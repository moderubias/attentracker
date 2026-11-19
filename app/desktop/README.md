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

1. Choose a camera index and target sampling period (default: 750ms).
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
a particular speedup. A truly fluid raw camera preview would require a
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
