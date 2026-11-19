# Attentracker

Local-first desktop activity observation prototype in Rust.

**Current functionality:** single-owner background camera capture (Nokhwa),
YuNet detection through Tract, last-frame-only inference, opt-in live local
preview with approximate recent face bounding boxes/landmarks, numerical
latency telemetry, and conservative, provisional activity states in
Tauri 2 + Svelte. On compatible Wayland compositors (ext-idle-notify-v1
version 2), input-idle *transitions*, not input contents, contribute
additional evidence during a monitoring session.

**Not implemented yet:** verified productivity assessment, robust
cross-platform OS activity acquisition, focus tracking, phone detection,
speaker or conversation detection, activity history, Kernix classifier
training, customizable charts, graphical export, background tray operation,
and packaged desktop installers. These remain planned, not current features.

## Architecture

- `crates/attent-camera`: RGB capture, metadata, letterbox preprocessing.
- `crates/attent-model`: ONNX inference and YuNet postprocessing.
- `crates/attent-core`: model/capture workers, one-slot latest-frame buffer,
  opt-in asynchronous preview, numerical observation snapshots, lifecycle.
- `crates/attent-activity`: temporal classifier and optional Wayland
  input-idle transition observer (Linux); unsupported platforms remain Unknown.
- `crates/attent-bin`: one-frame diagnostic command.
- `app/desktop`: optional Tauri 2 desktop prototype (separate Cargo workspace).

## Throughput experiments

Measured in the original sequential pipeline on the reference i3 laptop:
about **20.9 analyzed FPS at 320px** and **29.5 FPS at 160px** (release,
preview disabled). Those figures are baselines, not guarantees for the new
concurrent camera implementation; it requires a fresh on-device benchmark.
See [performance.md](docs/performance.md) and
[live-preview-activity.md](docs/live-preview-activity.md).

## Run

```sh
cargo fmt --all --check
cargo check --workspace
cargo test --workspace
cargo run -p attent-bin
```

For the graphical window on Linux, install Tauri 2 system dependencies
(including WebKitGTK 4.1). Then:

```sh
cd app/desktop
pnpm install
pnpm tauri dev
```

Desktop packaging is not supported yet. The development binary requires the
fixed or dynamic YuNet ONNX model to be available locally; use
`ATTENT_MODEL_PATH` to provide an explicit file path. A production Tauri
binary can be compiled on another Linux machine with
`cargo build --release --features tauri/custom-protocol`; see
[remote-build.md](docs/remote-build.md).

Camera device indices may vary by OS. The desktop defaults to index 1
as on the original development laptop, but it is configurable in the UI.

## Privacy and interpretation

Camera and optional Wayland input-idle monitoring begin only when the
user presses Start. Without preview, the UI receives only numerical data.
When explicitly enabled, fresh camera images are encoded in memory for the
**local** WebView. Face overlays are based on the most recent completed ML
observation and can be slightly older than the displayed image; the UI
labels their age. Disabling preview clears its image buffer; no image
uploads, remote telemetry or video files are implemented.

A face in view is **not** proof of work or productivity. The provisional
`Away` state means only that no face has been detected for eight seconds;
it is not proof of physical absence. `Working` means recent verified OS
input with one visible face, not task relevance or identity. Without input
evidence, `Working` and `Idling` cannot be assigned. Camera errors give
`Unknown`; input content and audio are never recorded.

This is early-stage software, not a security guarantee. Review camera
permissions, platform behavior and dependencies before use.

## Model and licensing

Project source is licensed under `MIT OR Apache-2.0`.
Bundled/third-party model weights and dependencies retain their own
licenses. Verify the specific YuNet model attribution and redistribution
terms before distributing packaged binaries.

Optional NanoDet-Plus and Silero VAD weights may be downloaded separately
with `bash scripts/fetch-optional-models.sh all`. They are **not loaded or
integrated** in the running application. See [experimental model notes](models/EXPERIMENTAL.md)
for provenance, integrity limitations, and integration requirements.

## Verification

The camera preprocessing and YuNet postprocessing have unit tests,
and the earlier one-frame CLI ran successfully on a Linux test machine.
The revised preview and timing instrumentation still require desktop
integration tests on the target machine.
