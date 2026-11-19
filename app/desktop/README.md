# Attentracker desktop prototype

Tauri 2 + Svelte 5, powered by `attent-core` (Rust). This prototype shows
face presence only. It does **not** infer working time, attention, or identity.

## Development

From the repository root:

```sh
cd app/desktop
pnpm install
pnpm tauri dev
```

Requires Rust and the Tauri 2 platform prerequisites (WebKitGTK 4.1 on Linux,
WebView2 on Windows). The checked-in `models/yunet.onnx` is loaded from the
repository root **in development**. Packaging and model-resource lookup are
intentionally not implemented yet; `bundle.active` is currently false.

The application opens no camera until the user presses **Start monitoring**.
The Rust worker returns only numerical snapshots to the UI. Stopping the
monitor is asynchronous and can take until the current capture/inference
operation completes. An errored camera remains `unknown`, not `away`.

`pnpm install` must resolve new frontend dependencies; the Tauri subproject
is a separate Cargo workspace. Run `cargo check` from `app/desktop/src-tauri`
to check the desktop Rust entrypoint.
