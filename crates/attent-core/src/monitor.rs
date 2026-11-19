use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError, Sender},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail, ensure};
use attent_camera::{AttentCamera, prepare_yunet};
use attent_model::{DetectorConfig, YuNetDetector, YuNetModel};
use nokhwa::utils::{CameraFormat, CameraIndex, FrameFormat, Resolution};
use serde::Serialize;
use tracing::{debug, error, info, trace, warn};

use crate::diagnostics::{DiagnosticsWindow, FrameTimings, scheduler_gap_ms};
use crate::preview::{FramePreview, encode_preview};

const MODEL_SIDE: u32 = 640;

#[derive(Clone, Debug)]
pub struct MonitorConfig {
    pub camera_index: u32,
    pub sample_interval: Duration,
    pub model_path: PathBuf,
}

impl MonitorConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.sample_interval <= Duration::from_secs(60),
            "sample_interval must be between 0 (unthrottled) and 60 s"
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
        self.preview_enabled.store(enabled, Ordering::Release);
        if !enabled {
            *slot = None;
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

        let handle = match thread::Builder::new()
            .name("attent-cv-monitor".into())
            .spawn(move || {
                let result = observe(config, &snapshot, &preview, &preview_enabled, &stop_rx);
                *preview.lock().unwrap_or_else(|p| p.into_inner()) = None;
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
                        state.face_count = None;
                    }
                    Err(err) => {
                        error!(
                            target: "attent_core::lifecycle",
                            error = %err,
                            frames_analyzed = state.frames_analyzed,
                            "monitor worker failed"
                        );
                        state.phase = MonitorPhase::Error;
                        state.presence = Presence::Unknown;
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
        Ok(())
    }
}

fn observe(
    config: MonitorConfig,
    shared: &Arc<Mutex<MonitorSnapshot>>,
    shared_preview: &Arc<Mutex<Option<FramePreview>>>,
    preview_enabled: &Arc<AtomicBool>,
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
    let model = YuNetModel::load(&config.model_path)
        .with_context(|| format!("loading YuNet at {}", config.model_path.display()))?;
    let model_load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
    let detector_config = DetectorConfig::default();
    info!(
        target: "attent_core::lifecycle",
        model_input_side = MODEL_SIDE,
        score_threshold = detector_config.score_threshold,
        nms_threshold = detector_config.nms_threshold,
        top_k = detector_config.top_k,
        max_faces = detector_config.max_faces,
        "detector parameters"
    );
    let detector = YuNetDetector::new(detector_config)?;

    let preferred = CameraFormat::new(Resolution::new(640, 480), FrameFormat::MJPEG, 30);
    let camera_open_started = Instant::now();
    let mut camera =
        AttentCamera::new_preferred(CameraIndex::Index(config.camera_index), preferred)
            .context("opening capture device")?;
    info!(
        target: "attent_core::lifecycle",
        model_load_ms,
        camera_open_ms = camera_open_started.elapsed().as_secs_f64() * 1000.0,
        camera_format = ?camera.format(),
        target_interval_ms = config.sample_interval.as_millis() as u64,
        "monitor initialized"
    );

    {
        let mut state = shared.lock().unwrap_or_else(|p| p.into_inner());
        if state.phase != MonitorPhase::Stopping {
            state.phase = MonitorPhase::Running;
        }
        state.camera_format = Some(format!("{:?}", camera.format()));
    }

    let mut previous_sample_start: Option<Instant> = None;
    let mut previous_receive_tick: Option<Instant> = None;
    let mut previous_driver_timestamp: Option<Duration> = None;
    let mut last_preview_at: Option<Instant> = None;
    let mut stats = DiagnosticsWindow::new();
    let mut previous_cycle: Option<(f64, f64)> = None;
    loop {
        // A stop request can arrive while the model/device is initializing.
        match stop_rx.try_recv() {
            Ok(()) | Err(mpsc::TryRecvError::Disconnected) => break,
            Err(mpsc::TryRecvError::Empty) => {}
        }

        let begun = Instant::now();
        let period_ms = previous_sample_start
            .map(|previous| begun.duration_since(previous).as_secs_f64() * 1000.0);
        previous_sample_start = Some(begun);

        let frame = camera.capture().context("capturing camera frame")?;
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
        let input = prepare_yunet(&frame, MODEL_SIDE).context("preprocessing frame")?;
        let preprocess_ms = preprocess_started.elapsed().as_secs_f64() * 1000.0;

        let inference_started = Instant::now();
        let raw = model.infer(&input).context("running YuNet")?;
        let inference_ms = inference_started.elapsed().as_secs_f64() * 1000.0;

        let postprocess_started = Instant::now();
        let faces = detector.detect(&raw).context("decoding YuNet detections")?;
        let postprocess_ms = postprocess_started.elapsed().as_secs_f64() * 1000.0;
        let analysis_ms = begun.elapsed().as_secs_f64() * 1000.0;

        // A stop request may have arrived during a long model inference.
        if !matches!(stop_rx.try_recv(), Err(mpsc::TryRecvError::Empty)) {
            break;
        }

        let mut preview_ms = None;
        let mut preview_error = None;
        let mut preview_encoded = false;
        let preview_requested = preview_enabled.load(Ordering::Relaxed)
            && last_preview_at.is_none_or(|t| t.elapsed() >= Duration::from_millis(300));
        if preview_requested {
            let preview_started = Instant::now();
            match encode_preview(&frame, &faces) {
                Ok(preview) => {
                    preview_encoded = true;
                    trace!(
                        target: "attent_core::preview",
                        sequence = preview.sequence,
                        encoded_bytes = preview.data_url.len(),
                        overlays = preview.faces.len(),
                        "JPEG preview encoded"
                    );
                    // Publish under the same lock used by opt-out; hiding the
                    // preview cannot race with publication of a later frame.
                    let mut slot = shared_preview.lock().unwrap_or_else(|p| p.into_inner());
                    if preview_enabled.load(Ordering::Acquire) {
                        *slot = Some(preview);
                    }
                }
                Err(err) => {
                    warn!(
                        target: "attent_core::preview",
                        error = %err,
                        "JPEG preview encoding failed"
                    );
                    preview_error = Some(err.to_string());
                },
            }
            preview_ms = Some(preview_started.elapsed().as_secs_f64() * 1000.0);
            last_preview_at = Some(Instant::now());
        }

        let snapshot_lock_started = Instant::now();
        let snapshot_lock_wait_ms;
        {
            let mut state = shared.lock().unwrap_or_else(|p| p.into_inner());
            snapshot_lock_wait_ms = snapshot_lock_started.elapsed().as_secs_f64() * 1000.0;
            state.frames_analyzed = state.frames_analyzed.saturating_add(1);
            state.presence = Presence::from_count(faces.len());
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
            state.last_preview_ms = preview_ms;
            state.last_observation_period_ms = period_ms;
            state.last_preview_error = preview_error;
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
            preview_encoded,
            preview_error: preview_error.is_some(),
            decode_ms: frame.meta.decode_duration.as_secs_f64() * 1000.0,
            preprocess_ms,
            inference_ms,
            postprocess_ms,
            preview_ms: preview_ms.unwrap_or(0.0),
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
            capture_ms = timings.capture_ms,
            receive_period_ms = timings.receive_period_ms.unwrap_or(0.0),
            post_capture_age_ms = timings.post_capture_age_ms,
            snapshot_lock_wait_ms = timings.snapshot_lock_wait_ms,
            driver_timestamp_available = frame.meta.driver_timestamp.is_some(),
            driver_timestamp_unchanged,
            preview_requested,
            preview_encoded,
            preview_error = timings.preview_error,
            decode_ms = timings.decode_ms,
            preprocess_ms = timings.preprocess_ms,
            inference_ms = timings.inference_ms,
            postprocess_ms = timings.postprocess_ms,
            preview_ms = timings.preview_ms,
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
        );
        match stop_rx.recv_timeout(remaining) {
            Ok(()) | Err(RecvTimeoutError::Disconnected) => break,
            Err(RecvTimeoutError::Timeout) => {}
        }
    }
    stats.report("stopped", config.sample_interval.as_millis() as u64, preview_enabled.load(Ordering::Relaxed));
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
    fn snapshot_initially_has_unknown_presence() {
        let monitor = Monitor::new();
        let s = monitor.snapshot();
        assert_eq!(s.phase, MonitorPhase::Stopped);
        assert_eq!(s.presence, Presence::Unknown);
        assert_eq!(s.face_count, None);
    }
}
