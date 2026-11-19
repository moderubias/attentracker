# Attentracker

Local-first desktop activity observation prototype in Rust.

**Current functionality:** a camera capture pipeline (Nokhwa), YuNet face
detection through Tract, landmark/bounding-box decoding, and a simple
asynchronous monitor that exposes *face presence* observations in a
Tauri 2 + Svelte desktop window.

**Not implemented yet:** working/idle classification, focus tracking,
phone detection, speaker or conversation detection, activity history,
Kernix classifier training, customizable charts, graphical export,
background tray operation, and packaged desktop installers. These are
planned features, not claims about what the program currently detects.

## Architecture

- `crates/attent-camera`: RGB capture, metadata, letterbox preprocessing.
- `crates/attent-model`: ONNX inference and YuNet postprocessing.
- `crates/attent-core`: start/stop monitor worker, bounded sampling,
  numerical observation snapshot, errors and monitoring state.
- `crates/attent-activity`: future OS activity sources (currently empty).
- `crates/attent-bin`: one-frame diagnostic command.
- `app/desktop`: optional Tauri 2 desktop prototype (separate Cargo workspace).

## Throughput experiments

The experimental desktop branch adds selectable 160/320/640 model
input, an asynchronous opt-in JPEG preview encoder, and a server-built
release benchmark procedure. **20 analyzed FPS has not yet been confirmed
on the target laptop.** See [performance.md](docs/performance.md).

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

Desktop packaging is not supported yet. The prototype locates the YuNet
model relative to the source tree and requires `models/yunet.onnx`.

Camera device indices may vary by OS. The desktop defaults to index 1
as on the original development laptop, but it is configurable in the UI.

## Privacy and interpretation

Camera access starts only after an explicit click. Without preview,
the desktop UI receives numerical observations only. When the user
explicitly enables diagnostic preview, sampled JPEG frames are encoded
in memory and transferred only to the local desktop WebView with synchronized
face bounding boxes and landmarks. Disabling preview clears the image
buffer; no video files or remote telemetry are implemented.

A face in view is **not** proof of work or productivity. No face is
**not** proof that a user is away. A camera error gives `unknown`,
not `away`. The current program does not record key content or audio.

This is early-stage software, not a security guarantee. Review camera
permissions, platform behavior and dependencies before use.

## Model and licensing

Project source is licensed under `MIT OR Apache-2.0`.
Bundled/third-party model weights and dependencies retain their own
licenses. Verify the specific YuNet model attribution and redistribution
terms before distributing packaged binaries.

## Verification

The camera preprocessing and YuNet postprocessing have unit tests,
and the earlier one-frame CLI ran successfully on a Linux test machine.
The revised preview and timing instrumentation still require desktop
integration tests on the target machine.
