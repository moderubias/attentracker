# Throughput benchmark and release build (Arch desktop + Debian 13 build server)

Performance target: **20 analyzed frames/s**, not 20 camera frames/s or
20 UI redraws/s. This is a target, not a measured guarantee. Reduced
input resolution can compromise small-face recall and landmark precision.

The working branch is `performance/yunet-input-and-preview`; it is
stacked on the diagnostics changes in PR #6. Compare identical conditions
with no other webcam consumer running. Benchmarks must use a **release**
binary because debug and release inference can differ materially.

## 1. Update the source on the Arch laptop

Run in a terminal *on the laptop*, not the VPS:

```sh
cd ~/All_Work_Folders/moderubias/projects/attentracker
git status --short
git fetch origin
git switch --track origin/performance/yunet-input-and-preview
pnpm --dir app/desktop install --frozen-lockfile
pnpm --dir app/desktop build
bash scripts/fetch-yunet-dynamic.sh
sha256sum models/yunet-dynamic.onnx
ssh kiyoromi 'mkdir -p ~/attentracker-build'
rsync -az --info=progress2 \
  --exclude='.git/' --exclude='target/' --exclude='node_modules/' \
  --exclude='.env' --exclude='.env.*' --exclude='.venv/' \
  ./ kiyoromi:~/attentracker-build/
```

If the local branch already exists, use
`git switch performance/yunet-input-and-preview` and
`git pull --ff-only` instead of `git switch --track`.
Save or commit local modifications *before switching*; do not discard
work. Do not exclude `app/desktop/dist/` from the rsync.

## 2. Compile once on the VPS, not on the laptop

Debian/Ubuntu prerequisite installation is documented in
[remote-build.md](remote-build.md). The earlier server was accessible as
`kiyoromi` with a Rust toolchain under `/root/.cargo`.
The following uses 2 concurrent Rust jobs to control peak RAM use:

```sh
ssh kiyoromi '
  set -eu
  . "$HOME/.cargo/env"
  cd "$HOME/attentracker-build"
  CLANG_LIB="$(find /usr/lib -name libclang.so -print -quit)"
  test -n "$CLANG_LIB"
  export LIBCLANG_PATH="$(dirname "$CLANG_LIB")"
  CARGO_BUILD_JOBS=2 cargo build --release \
    --features tauri/custom-protocol \
    --manifest-path app/desktop/src-tauri/Cargo.toml
'
```

Keep server `target/` for incremental rebuilding. Never run Cargo
locally if avoiding the laptop compilation load is the goal.

**Why `--features tauri/custom-protocol` is necessary:** plain
`cargo build --release` does not activate Tauri's production asset
protocol. Its binary can retain the `devUrl` from
`tauri.conf.json` and show "Could not connect to 127.0.0.1" when
Vite is absent. Tauri CLI `tauri build` activates this production
feature automatically; when using Cargo directly, supply it explicitly.
A `release` profile alone does not select the production asset origin.
Do not add `tauri/custom-protocol` as an unconditional dependency
feature, because that can interfere with `tauri dev`.

This performance branch may have stale checked-in Cargo lockfiles,
so this particular command deliberately omits `--locked`. If Cargo
resolves the lockfile, keep its update on the server until the
project's lockfiles are reconciled. The benchmark uses no extra
third-party libraries compared with the existing source.

## 3. Transfer the release executable and run it without Vite

```sh
cd ~/All_Work_Folders/moderubias/projects/attentracker
mkdir -p app/desktop/src-tauri/target/release
rsync -av --progress \
  kiyoromi:~/attentracker-build/app/desktop/src-tauri/target/release/attentracker-desktop \
  app/desktop/src-tauri/target/release/
ldd app/desktop/src-tauri/target/release/attentracker-desktop | grep 'not found' || true
```

When compiled with `tauri/custom-protocol`, the release Tauri
executable embeds the static Vite-built UI. The binary requires **no
running Vite process**, although Linux must have compatible
WebKitGTK/GTK system libraries. If `ldd` reports missing libraries,
resolve that before benchmarking.

```sh
ATTENT_MODEL_PATH="$PWD/models/yunet-dynamic.onnx" \
RUST_LOG='warn,attentracker_desktop=info,attent_core=info,attent_camera=info,attent_model=info' \
  ./app/desktop/src-tauri/target/release/attentracker-desktop \
  2>&1 | tee "$HOME/attentracker-release-benchmark.log"
```

The models are **not yet bundled** for public release. For this developer
benchmark set `ATTENT_MODEL_PATH` to the separately downloaded
`models/yunet-dynamic.onnx` file. The original fixed-640
`models/yunet.onnx` cannot be made into a 160/320 input graph by
changing just the Tract input fact; its internal shapes remain fixed
at 640. The verified dynamic-input `2026may` variant is published by
OpenCV Zoo under the YuNet model's MIT license.

**640px performance regression:** on the reference i3, the dynamic 640
variant required approximately 2.07 s per inference (0.43 analyzed FPS),
compared with approximately 160 ms in the original fixed 640 graph
(5.70 FPS). The app therefore automatically prefers the **sibling
`yunet.onnx` file for 640**, if `ATTENT_MODEL_PATH` names
`yunet-dynamic.onnx` and the fixed file is present. 160 and 320 keep
the dynamic model. This automatic selection applies after rebuilding
from this branch, and the chosen model path is included in startup
logs. A custom model filename is never overridden.

To reproduce the fixed model's 640 result with an **older binary** (no
rebuild), close the app and relaunch it with:
`ATTENT_MODEL_PATH="$PWD/models/yunet.onnx"`, then choose 640 in
the UI. Restore the dynamic path for the 160/320 tests.

`scripts/fetch-yunet-dynamic.sh` downloads the actual Git LFS object
from OpenCV's repository and checks the pinned SHA-256. Do not use the
131-byte Git LFS pointer as the model. A checksum mismatch is a hard
error; investigate rather than removing the check.

**Known failure in previous experiments:** starting 160 or 320 with
`ATTENT_MODEL_PATH=.../models/yunet.onnx` fails before camera
initialization because the fixed model's first convolution expects
640x640. The command below now selects the dynamic variant.

## 4. Reproducible test matrix

With the webcam at the same position and similar illumination, select
`Max throughput` (0 ms), then run each of these for at least 30 seconds
after the model has initialized:

| YuNet input | Preview | What it tests |
| --- | --- | --- |
| 640 | off | Original resolution in release mode |
| 320 | off | Primary candidate for 20 analyzed FPS |
| 160 | off | Lower-accuracy speed comparison |
| 320 | on | Off-thread JPEG overhead and UI responsiveness |

Stop between tests, change the input side and restart monitoring.
For a fair comparison, avoid verbose `trace` logging while timing.
Include face-present and face-absent scenes, and verify the overlay
coordinates; faster results are not useful if recall becomes inadequate.

```sh
rg 'event="monitor_window"|event="preview_window"|ERROR|WARN' \
   "$HOME/attentracker-release-benchmark.log" | tail -n 100

python scripts/throughput-report.py "$HOME/attentracker-release-benchmark.log"
```

The script reports **aggregate analyzed FPS** as
`sum(frames) / sum(elapsed_s)` by model size, preview and sampling period.
It does not measure raw camera FPS. Compare several windows rather than
a single instantaneous FPS label.

If the 320 or 160 variant cannot compile or execute the ONNX graph,
the error will appear under `monitor worker failed`. Restore 640
to retain the known functional baseline and report the complete error.
The CI model integration test downloads and verifies the dynamic
model, then executes 160, 320 and 640 input shapes with Tract.
A successful CI test is required before treating smaller resolutions
as supported. The original static 640 model remains a fallback.

## 5. Interpret the results

At 20 analyzed FPS, the system has a 50 ms average time budget per
completed sample. Capture, JPEG decode, preprocessing, ONNX execution,
postprocessing, thread scheduling and image copies all count.
An inference average of 45 ms does **not** prove 20 FPS unless
total observed throughput is at least 20 FPS.

The on-device i3 throughput must be measured; Linux CI on a faster
runner is only evidence of correct compilation and model execution.
