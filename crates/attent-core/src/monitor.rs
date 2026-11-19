use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError, Sender},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail, ensure};
use attent_activity::{ActivityClassifier, ActivityState, InputEvidence, InputIdleMonitor};
use attent_camera::prepare_yunet;
use attent_model::{DetectorConfig, YuNetDetector, YuNetModel};
use nokhwa::utils::{CameraFormat, CameraIndex, FrameFormat, Resolution};
use serde::Serialize;
use tracing::{debug, error, info, trace, warn};

use crate::capture_worker::{CaptureWorker, DetectionSample, SharedDetections};
use crate::diagnostics::{DiagnosticsWindow, FrameTimings, scheduler_gap_ms};
use crate::preview::FramePreview;
use crate::preview_worker::PreviewWorker;

#[derive(Clone, Debug)]
pub struct MonitorConfig {
    pub camera_index: u32,
    pub sample_interval: Duration,
    pub model_input_side: u32,
    pub model_path: PathBuf,
}

impl MonitorConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.sample_interval <= Duration::from_secs(60),
            "sample_interval must be between 0 (unthrottled) and 60 s"
        );
        ensure!(
            matches!(self.model_input_side, 160 | 320 | 640),
            "model_input_side must be 160, 320, or 640"
        );
        ensure!(
            !self.model_path.as_os_str().is_empty(),
            "model path is empty"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MonitorPhase {
    #[default]
    Stopped,
    Starting,
    Running,
    Stopping,
    Error,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Presence {
    #[default]
    Unknown,
    NoFace,
    FacePresent,
    MultipleFaces,
}

impl Presence {
    fn from_count(n: usize) -> Self {
        match n {
            0 => Self::NoFace,
            1 => Self::FacePresent,
            _ => Self::MultipleFaces,
        }
    }
}

/// No images or raw key events are retained in this state.
#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MonitorSnapshot {
    pub phase: MonitorPhase,
    pub presence: Presence,
    pub activity_state: ActivityState,
    /// Missing means no OS-wide input evidence; never assume activity.
    pub input_source: Option<String>,
    pub camera_index: Option<u32>,
    pub frames_analyzed: u64,
    pub face_count: Option<usize>,
    pub best_confidence: Option<f32>,
    pub last_observed_at_ms: Option<u64>,
    pub last_analysis_ms: Option<f64>,
    pub last_capture_ms: Option<f64>,
    pub last_decode_ms: Option<f64>,
    pub last_preprocess_ms: Option<f64>,
    pub last_inference_ms: Option<f64>,
    pub last_postprocess_ms: Option<f64>,
    pub last_preview_ms: Option<f64>,
    /// Time between the beginnings of two sampled observations.
    pub last_observation_period_ms: Option<f64>,
    pub camera_format: Option<String>,
    pub last_preview_error: Option<String>,
    pub last_error: Option<String>,
}

struct Worker {
    stop: Sender<()>,
    handle: JoinHandle<()>,
}

/// Owns at most one monitoring thread. No camera access happens before start().
pub struct Monitor {
    snapshot: Arc<Mutex<MonitorSnapshot>>,
    preview: Arc<Mutex<Option<FramePreview>>>,
    preview_enabled: Arc<AtomicBool>,
    preview_generation: Arc<AtomicU64>,
    worker: Mutex<Option<Worker>>,
}

impl Default for Monitor {
    fn default() -> Self {
        Self::new()
    }
}

impl Monitor {
    pub fn new() -> Self {
        Self {
            snapshot: Arc::new(Mutex::new(MonitorSnapshot::default())),
            preview: Arc::new(Mutex::new(None)),
            preview_enabled: Arc::new(AtomicBool::new(false)),
            preview_generation: Arc::new(AtomicU64::new(0)),
            worker: Mutex::new(None),
        }
    }

    pub fn snapshot(&self) -> MonitorSnapshot {
        self.snapshot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Returns a preview only if the user explicitly enabled it and it is newer.
    pub fn preview(&self, after_sequence: u64) -> Option<FramePreview> {
        if !self.preview_enabled.load(Ordering::Relaxed) {
            return None;
        }
        let result = self.preview
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .filter(|frame| frame.sequence > after_sequence)
            .cloned();
        if let Some(ref frame) = result {
            trace!(
                target: "attent_core::preview",
                sequence = frame.sequence,
                encoded_bytes = frame.data_url.len(),
                "local UI requested updated preview"
            );
        }
        result
    }

    /// Explicitly opt into sending sampled camera images to this local UI.
    /// Disabling immediately drops the compressed frame retained in memory.
    pub fn set_preview_enabled(&self, enabled: bool) {
        let mut slot = self.preview.lock().unwrap_or_else(|p| p.into_inner());
        self.preview_generation.fetch_add(1, Ordering::AcqRel);
        self.preview_enabled.store(enabled, Ordering::Release);
        if !enabled {
            *slot = None;
            let mut state = self.snapshot.lock().unwrap_or_else(|p| p.into_inner());
            state.last_preview_ms = None;
            state.last_preview_error = None;
        }
        info!(
            target: "attent_core::lifecycle",
            enabled,
            "diagnostic preview toggled; no images included in logs"
        );
    }

    /// Starts asynchronously. Device and model errors are reported via snapshot().
    pub fn start(&self, config: MonitorConfig) -> Result<()> {
        config.validate()?;
        info!(
            target: "attent_core::lifecycle",
            camera_index = config.camera_index,
            target_interval_ms = config.sample_interval.as_millis() as u64,
            model_input_side = config.model_input_side,
            "monitor start requested"
        );
        let mut worker = self
            .worker
            .lock()
            .map_err(|_| anyhow!("monitor worker lock is poisoned"))?;

        if worker.as_ref().is_some_and(|w| !w.handle.is_finished()) {
            warn!(target: "attent_core::lifecycle", "monitor already running or stopping");
            bail!("monitor is already running or stopping");
        }
        if let Some(completed) = worker.take() {
            completed
                .handle
                .join()
                .map_err(|_| anyhow!("previous monitor worker panicked"))?;
        }

        {
            let mut status = self.snapshot.lock().unwrap_or_else(|p| p.into_inner());
            *status = MonitorSnapshot {
                phase: MonitorPhase::Starting,
                camera_index: Some(config.camera_index),
                ..MonitorSnapshot::default()
            };
        }

        *self.preview.lock().unwrap_or_else(|p| p.into_inner()) = None;
        let (stop_tx, stop_rx) = mpsc::channel();
        let snapshot = Arc::clone(&self.snapshot);
        let preview = Arc::clone(&self.preview);
        let preview_enabled = Arc::clone(&self.preview_enabled);
        let preview_generation = Arc::clone(&self.preview_generation);

        let handle = match thread::Builder::new()
            .name("attent-cv-monitor".into())
            .spawn(move || {
                let result = observe(config, &snapshot, &preview, &preview_enabled, &preview_generation, &stop_rx);
                // End-of-session is a privacy boundary even on camera errors.
                // Holding this lock fences against an in-flight encoder publish.
                {
                    let mut slot = preview.lock().unwrap_or_else(|p| p.into_inner());
                    preview_generation.fetch_add(1, Ordering::AcqRel);
                    preview_enabled.store(false, Ordering::Release);
                    *slot = None;
                }
                let mut state = snapshot.lock().unwrap_or_else(|p| p.into_inner());
                match result {
                    Ok(()) => {
                        info!(
                            target: "attent_core::lifecycle",
                            frames_analyzed = state.frames_analyzed,
                            "monitor worker stopped"
                        );
                        state.phase = MonitorPhase::Stopped;
                        state.presence = Presence::Unknown;
                        state.activity_state = ActivityState::Unknown;
                        state.input_source = None;
                        state.face_count = None;
                    }
                    Err(err) => {
                        error!(
                            target: "attent_core::lifecycle",
                            error = %format!("{err:#}"),
                            frames_analyzed = state.frames_analyzed,
                            "monitor worker failed"
                        );
                        state.phase = MonitorPhase::Error;
                        state.presence = Presence::Unknown;
                        state.activity_state = ActivityState::Unknown;
                        state.input_source = None;
                        state.face_count = None;
                        state.last_error = Some(format!("{err:#}"));
                    }
                }
            }) {
            Ok(handle) => handle,
            Err(err) => {
                let mut state = self.snapshot.lock().unwrap_or_else(|p| p.into_inner());
                state.phase = MonitorPhase::Error;
                state.last_error = Some(err.to_string());
                return Err(err.into());
            }
        };

        *worker = Some(Worker {
            stop: stop_tx,
            handle,
        });
        info!(target: "attent_core::lifecycle", "monitor worker thread spawned");
        Ok(())
    }

    /// Non-blocking stop request; observe phase=stopped to confirm completion.
    pub fn stop(&self) -> Result<()> {
        let worker = self
            .worker
            .lock()
            .map_err(|_| anyhow!("monitor worker lock is poisoned"))?;

        if let Some(worker) = worker.as_ref() {
            if !worker.handle.is_finished() {
                {
                    let mut state = self.snapshot.lock().unwrap_or_else(|p| p.into_inner());
                    state.phase = MonitorPhase::Stopping;
                }
                info!(target: "attent_core::lifecycle", "monitor stop signal sent");
                let _ = worker.stop.send(());
            }
        }
        // An explicit Stop revokes preview consent immediately, not just after
        // the next camera frame or worker shutdown.
        drop(worker);
        self.set_preview_enabled(false);
        Ok(())
    }
}

fn observe(
    config: MonitorConfig,
    shared: &Arc<Mutex<MonitorSnapshot>>,
    shared_preview: &Arc<Mutex<Option<FramePreview>>>,
    preview_enabled: &Arc<AtomicBool>,
    preview_generation: &Arc<AtomicU64>,
    stop_rx: &Receiver<()>,
) -> Result<()> {
    if stop_rx.try_recv().is_ok() {
        return Ok(());
    }

    info!(
        target: "attent_core::lifecycle",
        "initializing YuNet and camera"
    );
    let load_started = Instant::now();
    let model = YuNetModel::load_for_side(&config.model_path, config.model_input_side)
        .with_context(|| format!(
            "loading YuNet at {} for {}x{} input; if changing input resolution, fetch the upstream dynamic model using bash scripts/fetch-yunet-dynamic.sh and set ATTENT_MODEL_PATH to models/yunet-dynamic.onnx",
            config.model_path.display(),
            config.model_input_side,
            config.model_input_side
        ))?;
    let model_load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
    let detector_config = DetectorConfig::default();
    info!(
        target: "attent_core::lifecycle",
        model_input_side = config.model_input_side,
        score_threshold = detector_config.score_threshold,
        nms_threshold = detector_config.nms_threshold,
        top_k = detector_config.top_k,
        max_faces = detector_config.max_faces,
        "detector parameters"
    );
    let detector = YuNetDetector::new(detector_config)?;

    let preview_worker = PreviewWorker::spawn(
        Arc::clone(shared_preview),
        Arc::clone(preview_enabled),
        Arc::clone(preview_generation),
        Arc::clone(shared),
    )?;
    let detections: SharedDetections = Arc::new(Mutex::new(None));
    let preferred = CameraFormat::new(Resolution::new(640, 480), FrameFormat::MJPEG, 30);
    let capture_worker = CaptureWorker::spawn(
        CameraIndex::Index(config.camera_index),
        preferred,
        preview_worker.sender(),
        Arc::clone(preview_enabled),
        Arc::clone(preview_generation),
        Arc::clone(&detections),
    )?;
    info!(
        target: "attent_core::lifecycle",
        model_load_ms,
        target_interval_ms = config.sample_interval.as_millis() as u64,
        "model initialized; camera capture and optional preview run independently"
    );
    {
        let mut state = shared.lock().unwrap_or_else(|p| p.into_inner());
        if state.phase != MonitorPhase::Stopping {
            state.phase = MonitorPhase::Running;
        }
    }

    let mut previous_sample_start: Option<Instant> = None;
    let mut previous_receive_tick: Option<Instant> = None;
    let mut previous_driver_timestamp: Option<Duration> = None;
    let mut last_analyzed_sequence = 0_u64;
    let mut stats = DiagnosticsWindow::new();
    let mut classifier = ActivityClassifier::new();
    let mut last_logged_activity = ActivityState::Unknown;
    // Start only after the user pressed Start. Drop stops the input observer
    // promptly and no keystrokes, coordinates or content are accessed.
    let input_observer = InputIdleMonitor::start();
    let mut previous_cycle: Option<(f64, f64)> = None;
    loop {
        // Wake periodically to honor Stop even if the camera is not delivering.
        if !matches!(stop_rx.try_recv(), Err(mpsc::TryRecvError::Empty)) {
            break;
        }
        let Some(frame) = capture_worker.next_after(
            last_analyzed_sequence, Duration::from_millis(100),
        )? else {
            continue;
        };
        let skipped = frame.meta.sequence.saturating_sub(last_analyzed_sequence.saturating_add(1));
        last_analyzed_sequence = frame.meta.sequence;

        let begun = Instant::now();
        let period_ms = previous_sample_start
            .map(|previous| begun.duration_since(previous).as_secs_f64() * 1000.0);
        previous_sample_start = Some(begun);
        if skipped != 0 {
            trace!(target: "attent_core::capture",
                skipped_frames = skipped, last_analyzed_sequence,
                "analyzer took latest camera frame; stale frames discarded");
        }
        {
            let mut state = shared.lock().unwrap_or_else(|p| p.into_inner());
            if state.camera_format.is_none() {
                state.camera_format = Some(format!("{:?}", frame.meta.stream_format));
            }
        }
        let receive_period_ms = previous_receive_tick.map(|previous| {
            frame.meta.received_tick.duration_since(previous).as_secs_f64() * 1000.0
        });
        previous_receive_tick = Some(frame.meta.received_tick);
        let driver_timestamp_unchanged = matches!(
            (previous_driver_timestamp, frame.meta.driver_timestamp),
            (Some(previous), Some(current)) if previous == current
        );
        if frame.meta.driver_timestamp.is_some() {
            previous_driver_timestamp = frame.meta.driver_timestamp;
        }
        let preprocess_started = Instant::now();
        let input = prepare_yunet(&frame, config.model_input_side).context("preprocessing frame")?;
        let preprocess_ms = preprocess_started.elapsed().as_secs_f64() * 1000.0;

        let inference_started = Instant::now();
        let raw = model.infer(&input).context("running YuNet")?;
        let inference_ms = inference_started.elapsed().as_secs_f64() * 1000.0;

        let postprocess_started = Instant::now();
        let faces = detector.detect(&raw).context("decoding YuNet detections")?;
        let postprocess_ms = postprocess_started.elapsed().as_secs_f64() * 1000.0;
        // Overlays are tagged with their true source frame; they can be applied
        // approximately to a newer live frame for up to 350 ms.
        *detections.lock().unwrap_or_else(|p| p.into_inner()) = Some(DetectionSample {
            sequence: frame.meta.sequence,
            observed_at: frame.meta.received_tick,
            faces: faces.clone(),
        });
        let input_idle_for = input_observer.idle_for();
        let activity_state = classifier.observe(
            Instant::now(),
            Some(faces.len()),
            InputEvidence { idle_for: input_idle_for },
        );
        if activity_state != last_logged_activity {
            info!(
                target: "attent_core::activity",
                event = "activity_transition",
                previous = ?last_logged_activity,
                current = ?activity_state,
                input_source = input_observer.source().unwrap_or("none"),
                input_idle_ms = input_idle_for.map(|duration| duration.as_millis() as u64),
                face_count = faces.len(),
                "provisional activity state changed; no raw input logged"
            );
            last_logged_activity = activity_state;
        }
        let analysis_ms = begun.elapsed().as_secs_f64() * 1000.0;

        // A stop request may have arrived during a long model inference.
        if !matches!(stop_rx.try_recv(), Err(mpsc::TryRecvError::Empty)) {
            break;
        }

        // Preview is dispatched independently by the capture thread; preview
        // events and JPEG timing appear in capture_window and preview_window.
        let preview_requested = false;
        let preview_enqueued = false;
        let preview_error_occurred = false;

        let snapshot_lock_started = Instant::now();
        let snapshot_lock_wait_ms;
        {
            let mut state = shared.lock().unwrap_or_else(|p| p.into_inner());
            snapshot_lock_wait_ms = snapshot_lock_started.elapsed().as_secs_f64() * 1000.0;
            state.frames_analyzed = state.frames_analyzed.saturating_add(1);
            state.presence = Presence::from_count(faces.len());
            state.activity_state = activity_state;
            state.input_source = input_observer.source().map(str::to_owned);
            state.face_count = Some(faces.len());
            state.best_confidence = faces.first().map(|f| f.confidence);
            state.last_observed_at_ms = frame
                .meta
                .received_at
                .duration_since(UNIX_EPOCH)
                .ok()
                .map(|d| d.as_millis().min(u64::MAX as u128) as u64);
            state.last_capture_ms = Some(frame.meta.capture_duration.as_secs_f64() * 1000.0);
            state.last_decode_ms = Some(frame.meta.decode_duration.as_secs_f64() * 1000.0);
            state.last_preprocess_ms = Some(preprocess_ms);
            state.last_inference_ms = Some(inference_ms);
            state.last_postprocess_ms = Some(postprocess_ms);
            // Actual JPEG encoding timing/errors are published by the preview worker.
            state.last_observation_period_ms = period_ms;
            state.last_analysis_ms = Some(analysis_ms);
            state.last_error = None;
        }
        let post_capture_age_ms = frame.meta.received_tick.elapsed().as_secs_f64() * 1000.0;

        // An interval of zero intentionally disables the artificial delay.
        // Inference still runs sequentially: this does not promise camera FPS.
        let iteration_ms = begun.elapsed().as_secs_f64() * 1000.0;
        let remaining = config.sample_interval.saturating_sub(begun.elapsed());
        let planned_sleep_ms = remaining.as_secs_f64() * 1000.0;
        let scheduler_gap_ms = scheduler_gap_ms(period_ms, previous_cycle);
        previous_cycle = Some((iteration_ms, planned_sleep_ms));
        let timings = FrameTimings {
            capture_ms: frame.meta.capture_duration.as_secs_f64() * 1000.0,
            receive_period_ms,
            post_capture_age_ms,
            snapshot_lock_wait_ms,
            driver_timestamp_unchanged,
            preview_requested,
            preview_enqueued,
            preview_error: preview_error_occurred,
            decode_ms: frame.meta.decode_duration.as_secs_f64() * 1000.0,
            preprocess_ms,
            inference_ms,
            postprocess_ms,
            preview_ms: 0.0,
            pipeline_ms: analysis_ms,
            iteration_ms,
            period_ms,
            planned_sleep_ms,
            scheduler_gap_ms,
        };

        debug!(
            target: "attent_core::pipeline",
            event = "frame",
            sequence = frame.meta.sequence,
            faces = faces.len(),
            activity_state = ?activity_state,
            capture_ms = timings.capture_ms,
            receive_period_ms = timings.receive_period_ms.unwrap_or(0.0),
            post_capture_age_ms = timings.post_capture_age_ms,
            snapshot_lock_wait_ms = timings.snapshot_lock_wait_ms,
            driver_timestamp_available = frame.meta.driver_timestamp.is_some(),
            driver_timestamp_unchanged,
            preview_requested,
            preview_enqueued,
            preview_error = timings.preview_error,
            decode_ms = timings.decode_ms,
            preprocess_ms = timings.preprocess_ms,
            inference_ms = timings.inference_ms,
            postprocess_ms = timings.postprocess_ms,
            preview_dispatch_ms = timings.preview_ms,
            skipped_capture_frames = skipped,
            pipeline_ms = timings.pipeline_ms,
            iteration_ms = timings.iteration_ms,
            observed_period_ms = timings.period_ms.unwrap_or(0.0),
            planned_sleep_ms,
            scheduler_gap_ms = timings.scheduler_gap_ms.unwrap_or(0.0),
            target_interval_ms = config.sample_interval.as_millis() as u64,
            "completed sampled frame"
        );
        stats.record(timings);
        stats.report_if_due(
            config.sample_interval.as_millis() as u64,
            preview_enabled.load(Ordering::Relaxed),
            config.model_input_side,
        );
        match stop_rx.recv_timeout(remaining) {
            Ok(()) | Err(RecvTimeoutError::Disconnected) => break,
            Err(RecvTimeoutError::Timeout) => {}
        }
    }
    stats.report(
        "stopped",
        config.sample_interval.as_millis() as u64,
        preview_enabled.load(Ordering::Relaxed),
        config.model_input_side,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presence_is_observational_not_productivity() {
        assert_eq!(Presence::from_count(0), Presence::NoFace);
        assert_eq!(Presence::from_count(1), Presence::FacePresent);
        assert_eq!(Presence::from_count(2), Presence::MultipleFaces);
        assert_eq!(Presence::from_count(8), Presence::MultipleFaces);
    }

    #[test]
    fn sampling_bounds_are_enforced() {
        let mut config = MonitorConfig {
            camera_index: 1,
            sample_interval: Duration::from_millis(250),
            model_input_side: 320,
            model_path: PathBuf::from("models/yunet.onnx"),
        };
        assert!(config.validate().is_ok());
        config.sample_interval = Duration::ZERO;
        assert!(config.validate().is_ok()); // run as fast as a single worker allows
        config.sample_interval = Duration::from_millis(50);
        assert!(config.validate().is_ok());
        config.sample_interval = Duration::from_secs(60);
        assert!(config.validate().is_ok());
        config.sample_interval = Duration::from_secs(61);
        assert!(config.validate().is_err());
        config.sample_interval = Duration::ZERO;
        config.model_input_side = 192;
        assert!(config.validate().is_err());
    }

    #[test]
    fn preview_is_disabled_by_default() {
        let monitor = Monitor::new();
        assert!(monitor.preview(0).is_none());
        monitor.set_preview_enabled(true);
        assert!(monitor.preview(0).is_none());
        monitor.set_preview_enabled(false);
        assert!(monitor.preview(0).is_none());
    }

    #[test]
    fn stop_revokes_preview_even_when_no_worker_is_active() {
        let monitor = Monitor::new();
        monitor.set_preview_enabled(true);
        assert!(monitor.preview_enabled.load(Ordering::Acquire));
        monitor.stop().unwrap();
        assert!(!monitor.preview_enabled.load(Ordering::Acquire));
        assert!(monitor.preview(0).is_none());
    }

    #[test]
    fn snapshot_initially_has_unknown_presence() {
        let monitor = Monitor::new();
        let s = monitor.snapshot();
        assert_eq!(s.phase, MonitorPhase::Stopped);
        assert_eq!(s.presence, Presence::Unknown);
        assert_eq!(s.activity_state, ActivityState::Unknown);
        assert_eq!(s.face_count, None);
    }
}
