#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
    path::PathBuf,
    sync::{Mutex, atomic::{AtomicU64, Ordering}},
    time::{Duration, Instant},
};

use attent_core::{FramePreview, Monitor, MonitorConfig, MonitorPhase, MonitorSnapshot};
use tauri::State;
use tracing::{error, info, trace, warn};
use tracing_subscriber::EnvFilter;

const STALL_THRESHOLD: Duration = Duration::from_secs(5);
static EXPECTED_INTERVAL_MS: AtomicU64 = AtomicU64::new(250);
struct StatusWatchdog {
    frames: u64,
    last_progress: Instant,
    last_warning: Option<Instant>,
}
static STATUS_WATCHDOG: Mutex<Option<StatusWatchdog>> = Mutex::new(None);

fn check_status_progress(snapshot: &MonitorSnapshot) {
    let Ok(mut guard) = STATUS_WATCHDOG.lock() else {
        return;
    };
    if snapshot.phase != MonitorPhase::Running {
        *guard = None;
        return;
    }
    let expected_interval_ms = EXPECTED_INTERVAL_MS.load(Ordering::Relaxed);
    let adaptive_threshold = STALL_THRESHOLD.max(
        Duration::from_millis(expected_interval_ms.saturating_mul(3)),
    );
    match guard.as_mut() {
        Some(previous) if previous.frames == snapshot.frames_analyzed => {
            if previous.last_progress.elapsed() >= adaptive_threshold
                && previous.last_warning.is_none_or(|t| t.elapsed() >= STALL_THRESHOLD)
            {
                previous.last_warning = Some(Instant::now());
                warn!(
                    target: "attentracker_desktop::watchdog",
                    frames_analyzed = snapshot.frames_analyzed,
                    no_progress_ms = previous.last_progress.elapsed().as_secs_f64() * 1000.0,
                    target_interval_ms = expected_interval_ms,
                    stall_threshold_ms = adaptive_threshold.as_millis() as u64,
                    "monitor reports running but has not published a new observation"
                );
            }
        }
        _ => {
            *guard = Some(StatusWatchdog {
                frames: snapshot.frames_analyzed,
                last_progress: Instant::now(),
                last_warning: None,
            });
        }
    }
}

/// Aggregate-only frontend metrics. Never contains images, pixels or camera payloads.
#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct FrontendWindow {
    polls: u32,
    busy_skips: u32,
    poll_errors: u32,
    preview_requests: u32,
    preview_updates: u32,
    preview_no_change: u32,
    preview_errors: u32,
    rendered_images: u32,
    status_latency_sum_ms: f64,
    status_latency_max_ms: f64,
    preview_latency_sum_ms: f64,
    preview_latency_max_ms: f64,
    preview_payload_chars: u64,
    last_received_sequence: u64,
    last_rendered_sequence: u64,
    preview_enabled: bool,
    phase: String,
}

#[tauri::command]
fn report_frontend_telemetry(report: FrontendWindow) {
    info!(
        target: "attentracker_desktop::frontend",
        event = "frontend_window",
        phase = %report.phase,
        preview_enabled = report.preview_enabled,
        polls = report.polls,
        busy_skips = report.busy_skips,
        poll_errors = report.poll_errors,
        preview_requests = report.preview_requests,
        preview_updates = report.preview_updates,
        preview_no_change = report.preview_no_change,
        preview_errors = report.preview_errors,
        rendered_images = report.rendered_images,
        status_ipc_avg_ms = report.status_latency_sum_ms / f64::from(report.polls.max(1)),
        status_ipc_max_ms = report.status_latency_max_ms,
        preview_ipc_avg_ms = report.preview_latency_sum_ms / f64::from(report.preview_requests.max(1)),
        preview_ipc_max_ms = report.preview_latency_max_ms,
        preview_payload_chars = report.preview_payload_chars,
        last_received_sequence = report.last_received_sequence,
        last_rendered_sequence = report.last_rendered_sequence,
        "frontend polling, image rendering and Tauri IPC summary"
    );
}

#[tauri::command]
fn monitor_status(monitor: State<'_, Monitor>) -> MonitorSnapshot {
    let snapshot = monitor.snapshot();
    check_status_progress(&snapshot);
    trace!(
        target: "attentracker_desktop::ipc",
        phase = ?snapshot.phase,
        frames_analyzed = snapshot.frames_analyzed,
        "monitor status requested"
    );
    snapshot
}

/// The OpenCV 2026may dynamic graph is very slow at 640px with Tract on
/// some CPUs. Prefer the original fixed graph for that mode, without
/// modifying the explicit dynamic model selection for 160/320px.
fn select_model_path_for_side(
    side: u32,
    configured: Option<PathBuf>,
    local_fixed: &std::path::Path,
    local_dynamic: &std::path::Path,
) -> PathBuf {
    if side == 640 {
        if let Some(ref candidate) = configured {
            if candidate.file_name() == Some(std::ffi::OsStr::new("yunet-dynamic.onnx")) {
                let sibling = candidate.with_file_name("yunet.onnx");
                if sibling.is_file() {
                    return sibling;
                }
            }
        } else if local_fixed.is_file() {
            return local_fixed.to_path_buf();
        }
    }
    configured.unwrap_or_else(|| {
        if local_dynamic.is_file() {
            local_dynamic.to_path_buf()
        } else if local_fixed.is_file() {
            local_fixed.to_path_buf()
        } else {
            // Retain source-checkout fallback used in the developer workflow.
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../models/yunet.onnx")
        }
    })
}

#[tauri::command]
fn start_monitor(
    camera_index: u32,
    interval_ms: u64,
    model_input_side: u32,
    monitor: State<'_, Monitor>,
) -> Result<(), String> {
    // A remote-built binary must not depend on the build host's source path.
    let local_fixed = std::path::Path::new("models/yunet.onnx");
    let local_dynamic = std::path::Path::new("models/yunet-dynamic.onnx");
    let configured = std::env::var_os("ATTENT_MODEL_PATH").map(PathBuf::from);
    let model_path = select_model_path_for_side(
        model_input_side, configured, local_fixed, local_dynamic
    );
    info!(
        target: "attentracker_desktop::ipc",
        model_input_side,
        model_path = %model_path.display(),
        "selected ONNX model"
    );
    info!(
        target: "attentracker_desktop::ipc",
        camera_index,
        interval_ms,
        model_input_side,
        "Start monitoring clicked"
    );
    match monitor.start(MonitorConfig {
        camera_index,
        sample_interval: Duration::from_millis(interval_ms),
        model_input_side,
        model_path,
    }) {
        Ok(()) => {
            EXPECTED_INTERVAL_MS.store(interval_ms, Ordering::Relaxed);
            Ok(())
        },
        Err(err) => {
            error!(target: "attentracker_desktop::ipc", error = %err, "start command failed");
            Err(err.to_string())
        }
    }
}

#[tauri::command]
fn stop_monitor(monitor: State<'_, Monitor>) -> Result<(), String> {
    info!(target: "attentracker_desktop::ipc", "Stop monitoring clicked");
    monitor.stop().map_err(|err| {
        error!(target: "attentracker_desktop::ipc", error = %err, "stop command failed");
        err.to_string()
    })
}

/// Camera images leave the Rust worker only after an explicit UI opt-in.
#[tauri::command]
fn set_preview_enabled(enabled: bool, monitor: State<'_, Monitor>) {
    monitor.set_preview_enabled(enabled);
}

#[tauri::command]
fn monitor_preview(after_sequence: u64, monitor: State<'_, Monitor>) -> Option<FramePreview> {
    let image = monitor.preview(after_sequence);
    trace!(
        target: "attentracker_desktop::ipc",
        after_sequence,
        returned_preview = image.is_some(),
        "preview polled"
    );
    if let Some(ref frame) = image {
        trace!(
            target: "attentracker_desktop::ipc",
            sequence = frame.sequence,
            payload_bytes = frame.data_url.len(),
            "preview returned to local UI"
        );
    }
    image
}

fn main() {
    // By default log lifecycle and 5-second aggregates, without per-frame spam.
    // RUST_LOG=attent_core=debug,attent_model=trace,... enables deeper diagnostics.
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new("warn,attentracker_desktop=info,attent_core=info,attent_camera=info,attent_model=info")
    });
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_thread_names(true)
        .with_ansi(false)
        .init();
    info!(
        target: "attentracker_desktop::lifecycle",
        version = env!("CARGO_PKG_VERSION"),
        platform = std::env::consts::OS,
        arch = std::env::consts::ARCH,
        debug_build = cfg!(debug_assertions),
        parallelism = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        "Attentracker desktop started; no camera images logged"
    );
    tauri::Builder::default()
        .manage(Monitor::new())
        .invoke_handler(tauri::generate_handler![
            monitor_status,
            start_monitor,
            stop_monitor,
            set_preview_enabled,
            monitor_preview,
            report_frontend_telemetry,
        ])
        .run(tauri::generate_context!())
        .expect("failed to launch Attentracker");
}

#[cfg(test)]
mod model_path_tests {
    use super::*;

    #[test]
    fn 320_keeps_explicit_dynamic_model() {
        let selected = select_model_path_for_side(
            320,
            Some(PathBuf::from("/models/yunet-dynamic.onnx")),
            std::path::Path::new("/models/yunet.onnx"),
            std::path::Path::new("/models/yunet-dynamic.onnx"),
        );
        assert_eq!(selected, PathBuf::from("/models/yunet-dynamic.onnx"));
    }

    #[test]
    fn 640_preserves_custom_explicit_model_path() {
        let selected = select_model_path_for_side(
            640,
            Some(PathBuf::from("/models/custom-experiment.onnx")),
            std::path::Path::new("/models/yunet.onnx"),
            std::path::Path::new("/models/yunet-dynamic.onnx"),
        );
        assert_eq!(selected, PathBuf::from("/models/custom-experiment.onnx"));
    }

    #[test]
    fn known_checked_in_fixed_model_is_preferred_at_640() {
        let fixed = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../models/yunet.onnx");
        assert!(fixed.is_file());
        let dynamic = fixed.with_file_name("yunet-dynamic.onnx");
        let selected = select_model_path_for_side(
            640, Some(dynamic), &fixed, &fixed.with_file_name("yunet-dynamic.onnx")
        );
        assert_eq!(selected, fixed);
    }
}
