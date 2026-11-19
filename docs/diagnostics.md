# Attentracker: diagnostic collection (Linux / Arch)

All diagnostics are **numeric**. Image pixels, face crops, JPEG contents, keyboard input and audio are never logged by the application. The optional camera preview remains in memory and only goes to the local WebView.

## Before running

From the checkout:

```bash
git status --short
git rev-parse --short HEAD
v4l2-ctl --list-devices
```

The index selected in the UI is a Nokhwa camera index. It is not guaranteed to match `/dev/videoN`, especially with virtual cameras.

## Session A: performance without high-volume logging

Use a normal info-level log first. Trace logs themselves affect throughput on slower systems.

```bash
cd app/desktop
pnpm install --frozen-lockfile
RUST_LOG='warn,attentracker_desktop=info,attent_core=info,attent_camera=info,attent_model=info' \
  pnpm tauri dev 2>&1 | tee "$HOME/attentracker-performance.log"
```

Use the **same camera, lighting and preview state**. Run and record: Balanced 250 ms for 30 s; Stop; Max 0 ms for 30 s; Stop. Repeat once. Then repeat with preview enabled. Do not run two copies of the program or another program that holds the camera.

The backend emits `attent_core::telemetry event=monitor_window` roughly every 5 s and the frontend emits `attentracker_desktop::frontend event=frontend_window` every 5 s. They cover different intervals and are not frame-by-frame synchronized.

## Session B: verbose root-cause tracing

```bash
RUST_LOG='warn,attentracker_desktop=trace,attent_core=debug,attent_camera=trace,attent_model=trace' \
  pnpm tauri dev 2>&1 | tee "$HOME/attentracker-trace.log"
```

Reproduce just one suspicious profile for 15–30 s, then Stop and exit. The trace can grow quickly and adds measurable overhead. Never compare raw FPS from a trace run with FPS from a less-verbose run.

## What to inspect

| Log target / field | Meaning |
| --- | --- |
| `attent_camera::device` | Requested vs negotiated capture format, fallback and camera-open time |
| `attent_camera::capture` | Backend wait, MJPEG-to-RGB decode, source format, driver timestamp |
| `attent_camera::preprocess` | Resize vs BGR/NCHW tensor preparation, letterbox dimensions |
| `attent_model::inference` | Tensor allocation/copy, Tract execution and ONNX output copy |
| `attent_model::detector` | Rejected/low-score candidates, NMS suppression and accepted faces |
| `attent_core::pipeline` | Per-pass timings, observation period, scheduler gap and timestamp continuity |
| `attent_core::telemetry` | 5 s FPS and mean/p95/max timing breakdown, preview successes/errors |
| `attentracker_desktop::frontend` | Poll frequency, IPC latency, preview updates vs actual image loads |
| `attentracker_desktop::watchdog` | Monitoring is running but no new numeric snapshot arrived after a grace period |

Notes:

- `receive_period_ms` measures the interval between **successive completed camera receives**, not native sensor FPS.
- `post_capture_age_ms` measures time **after the backend has delivered the frame** until its numeric result is published. It does not estimate camera exposure or sensor-to-host latency.
- `unchanged_driver_timestamps` reports timestamp equality when the backend supplies timestamps; equality is **not proof** that pixel data are duplicated.
- `preview_payload_chars` counts Base64 data-URL characters transferred to the local UI, not image file bytes on disk.
- `rendered_images` counts browser image-load events; a successful image load does not prove that the display was painted on time.
- `scheduler_gap_ms` is the positive part of observed start-to-start period minus the prior cycle duration and requested sleep. It can include OS scheduling, clock granularity and other unmeasured overhead.
- Camera waits and an essentially single-threaded inference loop can leave overall CPU percentage low; compare against **one logical CPU** and actual inference duration before assuming unused processing capacity.

## Optional system measurements

For Arch Linux:

```bash
sudo pacman -S --needed v4l-utils sysstat
v4l2-ctl --list-devices
pidstat -u -r -d -p ALL 1
```

`pidstat -p ALL` includes other processes and can expose their names. Prefer running it locally for observation rather than sharing its complete output. The process of interest is the Tauri/Rust binary `attentracker-desktop`.

## Verification commands

```bash
# From the repository root:
cargo fmt --all --check
cargo test --workspace
cd app/desktop
pnpm build
cd src-tauri
cargo fmt --check
cargo check
```

If a test fails, collect the **first** error and the full compiler diagnostics rather than editing dependencies blindly.

## Sharing logs safely

Read logs before sharing. They may contain home paths, camera identifiers, shell environment details, and other processes' messages from dependencies or tooling. Replace secrets, personally identifying paths and device IDs as necessary. Do not upload webcam screenshots, environment files, or the source capture without deliberate consent.
