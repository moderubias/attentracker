#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{path::PathBuf, time::Duration};

use attent_core::{FramePreview, Monitor, MonitorConfig, MonitorSnapshot};
use tauri::State;

#[tauri::command]
fn monitor_status(monitor: State<'_, Monitor>) -> MonitorSnapshot {
    monitor.snapshot()
}

#[tauri::command]
fn start_monitor(
    camera_index: u32,
    interval_ms: u64,
    monitor: State<'_, Monitor>,
) -> Result<(), String> {
    let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../models/yunet.onnx");
    monitor
        .start(MonitorConfig {
            camera_index,
            sample_interval: Duration::from_millis(interval_ms),
            model_path,
        })
        .map_err(|err| err.to_string())
}

#[tauri::command]
fn stop_monitor(monitor: State<'_, Monitor>) -> Result<(), String> {
    monitor.stop().map_err(|err| err.to_string())
}

/// Camera images leave the Rust worker only after an explicit UI opt-in.
#[tauri::command]
fn set_preview_enabled(enabled: bool, monitor: State<'_, Monitor>) {
    monitor.set_preview_enabled(enabled);
}

#[tauri::command]
fn monitor_preview(after_sequence: u64, monitor: State<'_, Monitor>) -> Option<FramePreview> {
    monitor.preview(after_sequence)
}

fn main() {
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
