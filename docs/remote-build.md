# Offload Linux Tauri compilation to a Debian/Ubuntu server

This document describes building the Rust backend on a Linux server and running
it locally on the developer's Arch Linux desktop. The development frontend
(Vite) remains on the local machine. Do not run `pnpm tauri dev` locally in
this workflow: that would start another local Cargo build.

**Compatibility conditions:** both machines must use the same CPU architecture
(e.g. x86_64), and the local machine must provide dynamic libraries required by
the server-built ELF. Building on an older/equal glibc baseline is generally
more portable than building on a newer one. Check `ldd` after transfer.

## 1. Server prerequisites

Here `kiyoromi` is an SSH alias that logs in as root on Debian/Ubuntu. If
your alias uses a non-root user, use `sudo` for `apt-get`. Do not interchange
SSH aliases that connect to different users or home directories.

```bash
ssh kiyoromi 'id -un; . "$HOME/.cargo/env"; rustc --version; cargo --version'
ssh kiyoromi 'apt-get update && apt-get install -y \
  build-essential pkg-config clang libclang-dev libv4l-dev \
  libwebkit2gtk-4.1-dev libayatana-appindicator3-dev \
  librsvg2-dev libssl-dev libxdo-dev patchelf nasm'
```

When `v4l2-sys-mit` fails with "Unable to find libclang", install
`libclang-dev`. In unusual LLVM layouts, set `LIBCLANG_PATH` to the
directory containing `libclang.so`.

## 2. Sync the source and prebuilt frontend

From the **local** checkout root, on the diagnostic branch:

```bash
git pull --ff-only
pnpm --dir app/desktop install --frozen-lockfile
pnpm --dir app/desktop build
ssh kiyoromi 'mkdir -p ~/attentracker-build'
rsync -az --info=progress2 \
  --exclude='.git/' --exclude='target/' --exclude='node_modules/' \
  --exclude='.env' --exclude='.env.*' --exclude='.venv/' \
  ./ kiyoromi:~/attentracker-build/
```

Do **not** exclude `app/desktop/dist/` in this workflow. The Tauri
build may need it during context generation.

## 3. Compile Rust remotely

```bash
ssh kiyoromi '
  set -eu
  . "$HOME/.cargo/env"
  cd "$HOME/attentracker-build"
  clang_lib="$(find /usr/lib -name libclang.so -print -quit)"
  if [ -z "$clang_lib" ]; then
    echo "libclang.so missing: install libclang-dev" >&2
    exit 1
  fi
  export LIBCLANG_PATH="$(dirname "$clang_lib")"
  CARGO_BUILD_JOBS=4 cargo build --locked \
    --manifest-path app/desktop/src-tauri/Cargo.toml
'
```

Use 2 jobs if available memory is scarce. Initial compilation is
substantial, but subsequent builds reuse the server's `target/` cache.

## 4. Transfer only the binary

```bash
mkdir -p app/desktop/src-tauri/target/debug
rsync -av --progress \
  kiyoromi:~/attentracker-build/app/desktop/src-tauri/target/debug/attentracker-desktop \
  app/desktop/src-tauri/target/debug/
ldd app/desktop/src-tauri/target/debug/attentracker-desktop | grep 'not found' || true
```

The `ldd` output should contain no "not found" entries. If there are
missing libraries, install compatible local Arch Linux packages.

## 5. Run on Arch without compiling Rust

In **terminal A** (from the checkout root):

```bash
pnpm --dir app/desktop dev
```

Leave Vite running. In **terminal B**, also from the checkout root:

```bash
ATTENT_MODEL_PATH="$PWD/models/yunet.onnx" \
RUST_LOG='warn,attentracker_desktop=trace,attent_core=debug,attent_camera=trace,attent_model=trace' \
  ./app/desktop/src-tauri/target/debug/attentracker-desktop \
  2>&1 | tee "$HOME/attentracker-remote-built-trace.log"
```

`ATTENT_MODEL_PATH` exists to override the compile-time model location,
which otherwise points to the **server's** source checkout. The file
`models/yunet.onnx` must exist locally.

The debug Tauri binary loads the Vite dev URL configured in
`app/desktop/src-tauri/tauri.conf.json`. Running the binary without Vite
will not serve the desktop UI. A release/bundled executable requires a
different build workflow.

If the binary cannot launch, collect its stderr and `ldd` output before
trying a different compiler or rebuilding the entire project locally.
