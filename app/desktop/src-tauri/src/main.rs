#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{path::PathBuf, time::Duration};

use attent_core::{FramePreview, Monitor, MonitorConfig, MonitorSnapshot};
use tauri::State;
use tracing::{error, info, trace};
use tracing_subscriber::EnvFilter;

#[tauri::command]
fn monitor_status(monitor: State<'_, Monitor>) -> MonitorSnapshot {
    let snapshot = monitor.snapshot();
    trace!(
        target: "attentracker_desktop::ipc",
        phase = ?snapshot.phase,
        frames_analyzed = snapshot.frames_analyzed,
        "monitor status requested"
    );
    snapshot
}

#[tauri::command]
fn start_monitor(
    camera_index: u32,
    interval_ms: u64,
    monitor: State<'_, Monitor>,
) -> Result<(), String> {
    let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../models/yunet.onnx");
    info!(
        target: "attentracker_desktop::ipc",
        camera_index,
        interval_ms,
        "Start monitoring clicked"
    );
    match monitor.start(MonitorConfig {
        camera_index,
        sample_interval: Duration::from_millis(interval_ms),
        model_path,
    }) {
        Ok(()) => Ok(()),
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
        ])
        .run(tauri::generate_context!())
        .expect("failed to launch Attentracker");
}
