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
            (Duration::from_millis(250)..=Duration::from_secs(60)).contains(&self.sample_interval),
            "sample_interval must be between 250 ms and 60 s"
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
        self.preview
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .filter(|frame| frame.sequence > after_sequence)
            .cloned()
    }

    /// Explicitly opt into sending sampled camera images to this local UI.
    /// Disabling immediately drops the compressed frame retained in memory.
    pub fn set_preview_enabled(&self, enabled: bool) {
        self.preview_enabled.store(enabled, Ordering::Relaxed);
        if !enabled {
            *self.preview.lock().unwrap_or_else(|p| p.into_inner()) = None;
        }
    }

    /// Starts asynchronously. Device and model errors are reported via snapshot().
    pub fn start(&self, config: MonitorConfig) -> Result<()> {
        config.validate()?;
        let mut worker = self
            .worker
            .lock()
            .map_err(|_| anyhow!("monitor worker lock is poisoned"))?;

        if worker.as_ref().is_some_and(|w| !w.handle.is_finished()) {
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
                        state.phase = MonitorPhase::Stopped;
                        state.presence = Presence::Unknown;
                        state.face_count = None;
                    }
                    Err(err) => {
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

    let model = YuNetModel::load(&config.model_path)
        .with_context(|| format!("loading YuNet at {}", config.model_path.display()))?;
    let detector = YuNetDetector::new(DetectorConfig::default())?;

    let preferred = CameraFormat::new(Resolution::new(640, 480), FrameFormat::MJPEG, 30);
    let mut camera =
        AttentCamera::new_preferred(CameraIndex::Index(config.camera_index), preferred)
            .context("opening capture device")?;

    {
        let mut state = shared.lock().unwrap_or_else(|p| p.into_inner());
        if state.phase != MonitorPhase::Stopping {
            state.phase = MonitorPhase::Running;
        }
        state.camera_format = Some(format!("{:?}", camera.format()));
    }

    let mut previous_sample_start: Option<Instant> = None;
    let mut last_preview_at: Option<Instant> = None;
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
        if preview_enabled.load(Ordering::Relaxed)
            && last_preview_at.is_none_or(|t| t.elapsed() >= Duration::from_millis(300))
        {
            let preview_started = Instant::now();
            match encode_preview(&frame, &faces) {
                Ok(preview) => {
                    // If disabled while encoding, don't publish the image.
                    if preview_enabled.load(Ordering::Relaxed) {
                        *shared_preview.lock().unwrap_or_else(|p| p.into_inner()) =
                            Some(preview);
                    }
                }
                Err(err) => preview_error = Some(err.to_string()),
            }
            preview_ms = Some(preview_started.elapsed().as_secs_f64() * 1000.0);
            last_preview_at = Some(Instant::now());
        }

        {
            let mut state = shared.lock().unwrap_or_else(|p| p.into_inner());
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

        let remaining = config.sample_interval.saturating_sub(begun.elapsed());
        match stop_rx.recv_timeout(remaining) {
            Ok(()) | Err(RecvTimeoutError::Disconnected) => break,
            Err(RecvTimeoutError::Timeout) => {}
        }
    }
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
        config.sample_interval = Duration::from_millis(249);
        assert!(config.validate().is_err());
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
