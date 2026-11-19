# Attentracker / Observation Studio

Tauri 2 + Svelte 5, powered by `attent-core` (Rust). This is a
**diagnostic prototype**: it detects faces and measures the capture,
preprocessing, inference and postprocessing pipeline. It does **not**
infer productivity, attention, user identity, or phone use.

## Run

```sh
cd app/desktop
pnpm install
pnpm tauri dev
```

Prerequisites: Rust, pnpm, and Tauri 2 platform libraries (WebKitGTK
4.1 on Linux, WebView2 on Windows). The ONNX model is currently loaded
from the source checkout `models/yunet.onnx`. Packaging is not yet
supported; `bundle.active = false`.

The Rust build script converts `assets/icons/icon.jpg` to an RGBA PNG
before the Tauri context is generated. The resulting PNG is ignored in Git.

## Using the diagnostics

1. Choose a camera index and sample period (default: 250ms). Use the Eco, Balanced, or Max throughput presets. Max throughput (0ms) disables artificial sleep between successive inference passes; this does not eliminate inference latency.
2. Press **Start monitoring**. The camera stays off before this action.
3. Press **Show camera preview** to opt into seeing sampled camera images.
   The preview displays *the exact frame used for the overlays*, so it
   updates with model inference rather than at hardware camera FPS.
4. Inspect bounding boxes, five landmarks and stage timings to see which
   operation is limiting the actual analysis frequency.
5. Press **Hide camera preview** to immediately drop the compressed
   thumbnail buffer; **Stop** releases the camera and worker.

**Privacy:** without preview, the UI receives only numerical observations.
With preview enabled, in-memory JPEG images are passed to the *local
Tauri WebView only*. No image file is written by the application and no
network upload or remote telemetry is implemented. Users can nevertheless
take screenshots; consider other people in the camera's field of view.

The project uses system fonts only and does not fetch remote UI assets.

## Diagnostic logging

All logs are written by the Rust backend to the terminal running Tauri.
The default logging level reports startup/shutdown, camera negotiation,
preview permission changes, failures, and one performance window approximately
every 5 seconds. These events contain numbers only, never images, keyboard
input, audio, or the JPEG/base64 preview.

For a **shareable verbose trace** on Linux, run from `app/desktop`:

```sh
RUST_LOG='warn,attentracker_desktop=info,attent_core=debug,attent_camera=trace,attent_model=trace' \
  pnpm tauri dev 2>&1 | tee "$HOME/attentracker-dai10.log"
```

`attent_core::pipeline` DEBUG entries report one observation per line.
`attent_core::telemetry` INFO entries summarize each ~5-second window:
effective FPS, mean/p50/p95/max inference time, camera/decode/preview
overhead, requested sleep, and any otherwise unaccounted scheduler gap.
`attent_model::inference` TRACE distinguishes tensor creation, actual
Tract execution, and output copies. `attent_camera::capture` TRACE
reports image resolution and capture/decode durations.

To compare scheduling fairly: use the same camera, lighting and preview
setting; run Balanced for at least 30 seconds, stop, then Max throughput
for 30 seconds. Repeat once. Window-to-window variation matters more than
a single instantaneous FPS number. Switch the camera preview OFF for
the first comparison, then ON for a second pair of trials.

**Before sharing the log:** inspect it for filesystem paths, camera device
identifiers, process/environment details and any incidental diagnostic
messages from dependencies. The app never intentionally logs raw camera
pixels or preview payloads, but other tools may print additional output.
There is no automatic upload.

## Performance interpretation

The main latency number measures camera wait + MJPEG decode + tensor prep
+ Tract inference + NMS, **not just ONNX inference**. Preview JPEG cost is
measured separately. Actual model observation FPS is the reciprocal of the
period between completed sampling passes, not the camera nominal FPS.

The Tauri development workspace enables `opt-level = 2` only for
selected CV and Tract crates, not the entire GTK/Tauri dependency tree.
This makes runtime measurements more representative than Rust's fully
unoptimized defaults, at the cost of extra compilation and disk usage.
Use the profiler to compare before and after results, rather than assume
a particular speedup. At 0ms, the sequential worker runs continuously; observed
FPS is bounded by the entire model-processing pass and may raise CPU usage. A truly fluid raw camera preview would require a
separate capture/preview loop and remains future work.

The Tauri subproject is a separate Cargo workspace. Check it separately:

```sh
cd app/desktop/src-tauri
cargo check
```

For the core workspace from the repository root:

```sh
cargo fmt --all --check
cargo check --workspace
cargo test --workspace
```

## Planned, not implemented

Activity source tracking, SQLite session history, Kernix temporal
classification, phone detection, charts, image export, and packaged
installers are out of scope for this diagnostic milestone.
