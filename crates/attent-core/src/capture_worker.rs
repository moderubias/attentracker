//! Single camera owner that continuously publishes its newest decoded frame.
//! Inference never accumulates a queue of stale video, and optional preview
//! receives independent samples without opening the camera twice.

use std::{
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::SyncSender,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use anyhow::{Result, anyhow, bail};
use attent_camera::{AttentCamera, CapturedFrame};
use attent_model::FaceDetection;
use nokhwa::utils::{CameraFormat, CameraIndex};
use tracing::{debug, info, trace, warn};

use crate::preview_worker::{PreviewJob, PreviewWorker, SubmitResult};

const PREVIEW_PERIOD: Duration = Duration::from_millis(85);
const MAX_OVERLAY_AGE: Duration = Duration::from_millis(350);

pub(crate) struct DetectionSample {
    pub sequence: u64,
    pub observed_at: Instant,
    pub faces: Vec<FaceDetection>,
}

pub(crate) type SharedDetections = Arc<Mutex<Option<DetectionSample>>>;

#[derive(Default)]
struct LatestFrame {
    frame: Option<Arc<CapturedFrame>>,
    error: Option<String>,
    finished: bool,
}

pub(crate) struct CaptureWorker {
    shared: Arc<(Mutex<LatestFrame>, Condvar)>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl CaptureWorker {
    pub(crate) fn spawn(
        index: CameraIndex,
        requested: CameraFormat,
        preview_sender: SyncSender<PreviewJob>,
        preview_enabled: Arc<AtomicBool>,
        generation: Arc<AtomicU64>,
        detections: SharedDetections,
    ) -> Result<Self> {
        let shared = Arc::new((Mutex::new(LatestFrame::default()), Condvar::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let latest = Arc::clone(&shared);
        let stopped = Arc::clone(&stop);
        let handle = thread::Builder::new()
            .name("attent-capture".to_owned())
            .spawn(move || {
                let result: Result<()> = (|| {
                    let open_started = Instant::now();
                    let mut camera = AttentCamera::new_preferred(index, requested)?;
                    info!(
                        target: "attent_core::capture",
                        camera_open_ms = open_started.elapsed().as_secs_f64() * 1000.0,
                        negotiated_format = ?camera.format(),
                        "independent camera worker initialized"
                    );
                    let mut last_preview = None::<Instant>;
                    let mut window_start = Instant::now();
                    let mut window_frames = 0_u64;
                    let mut preview_queued = 0_u64;
                    let mut preview_dropped = 0_u64;
                    loop {
                        if stopped.load(Ordering::Acquire) {
                            break;
                        }
                        let frame = Arc::new(camera.capture()?);
                        window_frames += 1;
                        // A single retained frame; slow inference skips old frames.
                        {
                            let (slot, wake) = &*latest;
                            let mut state = slot.lock().unwrap_or_else(|e| e.into_inner());
                            state.frame = Some(Arc::clone(&frame));
                            wake.notify_all();
                        }

                        if preview_enabled.load(Ordering::Acquire)
                            && last_preview.is_none_or(|previous| previous.elapsed() >= PREVIEW_PERIOD)
                        {
                            last_preview = Some(Instant::now());
                            let sample = {
                                let guard = detections.lock().unwrap_or_else(|e| e.into_inner());
                                guard.as_ref().and_then(|s| {
                                    let age = frame.meta.received_tick
                                        .saturating_duration_since(s.observed_at);
                                    if s.sequence <= frame.meta.sequence && age <= MAX_OVERLAY_AGE {
                                        Some((s.sequence, age.as_secs_f64() * 1000.0, s.faces.clone()))
                                    } else {
                                        None
                                    }
                                })
                            };
                            let (detection_sequence, detection_age_ms, faces) = match sample {
                                Some((sequence, age, faces)) => (Some(sequence), Some(age), faces),
                                None => (None, None, Vec::new()),
                            };
                            let job = PreviewJob {
                                frame: Arc::clone(&frame),
                                faces,
                                detection_sequence,
                                detection_age_ms,
                                generation: generation.load(Ordering::Acquire),
                            };
                            match PreviewWorker::submit(&preview_sender, job) {
                                Ok(SubmitResult::Queued) => preview_queued += 1,
                                Ok(SubmitResult::Full) => preview_dropped += 1,
                                Err(err) => {
                                    warn!(target: "attent_core::capture",
                                        error = %err, "preview encoder disconnected");
                                    // Capture and inference remain operational without preview.
                                }
                            }
                        }

                        if window_start.elapsed() >= Duration::from_secs(5) {
                            info!(
                                target: "attent_core::capture",
                                event = "capture_window",
                                captured = window_frames,
                                capture_fps = window_frames as f64 / window_start.elapsed().as_secs_f64(),
                                preview_queued,
                                preview_dropped,
                                "live camera frame producer summary"
                            );
                            window_start = Instant::now();
                            window_frames = 0;
                            preview_queued = 0;
                            preview_dropped = 0;
                        }
                        trace!(target: "attent_core::capture",
                            sequence = frame.meta.sequence,
                            "latest frame published");
                    }
                    debug!(target: "attent_core::capture", "camera worker stopped");
                    Ok(())
                })();
                let (slot, wake) = &*latest;
                let mut state = slot.lock().unwrap_or_else(|e| e.into_inner());
                if let Err(err) = result {
                    warn!(target: "attent_core::capture", error = %format!("{err:#}"),
                        "camera worker exited with error");
                    state.error = Some(format!("{err:#}"));
                }
                state.finished = true;
                wake.notify_all();
            })?;
        Ok(Self { shared, stop, handle: Some(handle) })
    }

    /// Wait for a frame newer than the last analyzed one. This intentionally
    /// skips old frames instead of replaying a backlog.
    pub(crate) fn next_after(
        &self,
        after_sequence: u64,
        timeout: Duration,
    ) -> Result<Option<Arc<CapturedFrame>>> {
        let (slot, wake) = &*self.shared;
        let mut state = slot.lock().unwrap_or_else(|e| e.into_inner());
        if state.frame.as_ref().is_none_or(|f| f.meta.sequence <= after_sequence)
            && state.error.is_none() && !state.finished
        {
            let (next, _) = wake.wait_timeout(state, timeout)
                .unwrap_or_else(|e| e.into_inner());
            state = next;
        }
        if let Some(err) = &state.error {
            bail!("camera producer failed: {err}");
        }
        if let Some(frame) = &state.frame {
            if frame.meta.sequence > after_sequence {
                return Ok(Some(Arc::clone(frame)));
            }
        }
        if state.finished {
            return Err(anyhow!("camera producer stopped unexpectedly"));
        }
        Ok(None)
    }
}

impl Drop for CaptureWorker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            if handle.join().is_err() {
                warn!(target: "attent_core::capture", "camera worker panicked");
            }
        }
    }
}
