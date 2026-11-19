//! Bounded, opt-in JPEG encoder off the latency-critical YuNet worker.
//! The channel holds at most one queued RGB frame; old work is dropped rather
//! than allowing preview latency and memory use to grow without bound.

use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::{self, SyncSender, TrySendError},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use anyhow::Result;
use attent_model::FaceDetection;
use image::RgbImage;
use tracing::{error, info, trace, warn};

use crate::{
    monitor::MonitorSnapshot,
    preview::{FramePreview, encode_preview_rgb},
};

pub(crate) struct PreviewJob {
    pub rgb: RgbImage,
    pub sequence: u64,
    pub faces: Vec<FaceDetection>,
    pub generation: u64,
}

pub(crate) enum SubmitResult {
    Queued,
    Full,
}

pub(crate) struct PreviewWorker {
    sender: Option<SyncSender<PreviewJob>>,
    handle: Option<JoinHandle<()>>,
}

impl PreviewWorker {
    pub(crate) fn spawn(
        shared: Arc<Mutex<Option<FramePreview>>>,
        enabled: Arc<AtomicBool>,
        generation: Arc<AtomicU64>,
        status: Arc<Mutex<MonitorSnapshot>>,
    ) -> Result<Self> {
        let (tx, rx) = mpsc::sync_channel::<PreviewJob>(1);
        let handle = thread::Builder::new()
            .name("attent-preview".to_owned())
            .spawn(move || {
                let window = Duration::from_secs(5);
                let mut started = Instant::now();
                let mut count = 0_u64;
                let mut errors = 0_u64;
                let mut sum_ms = 0.0_f64;
                let mut max_ms = 0.0_f64;

                let report = |reason: &str, count: &mut u64, errors: &mut u64,
                              sum_ms: &mut f64, max_ms: &mut f64, started: &mut Instant| {
                    if *count + *errors > 0 {
                        info!(
                            target: "attent_core::preview",
                            event = "preview_window",
                            reason,
                            encoded = *count,
                            errors = *errors,
                            encode_avg_ms = *sum_ms / (*count).max(1) as f64,
                            encode_max_ms = *max_ms,
                            elapsed_s = started.elapsed().as_secs_f64(),
                            "asynchronous JPEG encoding summary"
                        );
                    }
                    *count = 0;
                    *errors = 0;
                    *sum_ms = 0.0;
                    *max_ms = 0.0;
                    *started = Instant::now();
                };

                for job in rx {
                    // Generation fences stale frames across disable/re-enable.
                    if !enabled.load(Ordering::Acquire)
                        || generation.load(Ordering::Acquire) != job.generation
                    {
                        continue;
                    }

                    let encode_started = Instant::now();
                    let outcome = encode_preview_rgb(&job.rgb, job.sequence, &job.faces);
                    let elapsed_ms = encode_started.elapsed().as_secs_f64() * 1000.0;
                    match outcome {
                        Ok(image) => {
                            trace!(
                                target: "attent_core::preview",
                                sequence = image.sequence,
                                encode_ms = elapsed_ms,
                                encoded_chars = image.data_url.len(),
                                "JPEG preview encoded off inference thread"
                            );
                            // Disable/enable holds the same preview mutex and
                            // increments generation to invalidate in-flight work.
                            let mut slot = shared.lock().unwrap_or_else(|p| p.into_inner());
                            if enabled.load(Ordering::Acquire)
                                && generation.load(Ordering::Acquire) == job.generation
                            {
                                *slot = Some(image);
                            }
                            drop(slot);
                            let mut state = status.lock().unwrap_or_else(|p| p.into_inner());
                            state.last_preview_ms = Some(elapsed_ms);
                            state.last_preview_error = None;
                            count += 1;
                            sum_ms += elapsed_ms;
                            max_ms = max_ms.max(elapsed_ms);
                        }
                        Err(err) => {
                            warn!(
                                target: "attent_core::preview",
                                sequence = job.sequence,
                                error = %err,
                                "asynchronous JPEG preview encoding failed"
                            );
                            let mut state = status.lock().unwrap_or_else(|p| p.into_inner());
                            state.last_preview_error = Some(err.to_string());
                            errors += 1;
                        }
                    }
                    if started.elapsed() >= window {
                        report("periodic", &mut count, &mut errors,
                               &mut sum_ms, &mut max_ms, &mut started);
                    }
                }
                report("stopped", &mut count, &mut errors,
                       &mut sum_ms, &mut max_ms, &mut started);
            })?;
        Ok(Self {
            sender: Some(tx),
            handle: Some(handle),
        })
    }

    pub(crate) fn submit(&self, job: PreviewJob) -> Result<SubmitResult> {
        match self.sender.as_ref().expect("preview sender present").try_send(job) {
            Ok(()) => Ok(SubmitResult::Queued),
            Err(TrySendError::Full(_)) => Ok(SubmitResult::Full),
            Err(TrySendError::Disconnected(_)) => {
                anyhow::bail!("preview encoder thread unexpectedly exited")
            }
        }
    }
}

impl Drop for PreviewWorker {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(handle) = self.handle.take() {
            if handle.join().is_err() {
                error!(target: "attent_core::preview", "preview encoder thread panicked");
            }
        }
    }
}
