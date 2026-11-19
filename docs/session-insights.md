# Ephemeral activity insights and appearance

The session timeline uses only numeric monitor snapshots. It does not assess
productivity, detect phones, process audio, or store permanent history.

- At most 60 minutes of activity state, timestamp, and presence category are
  held in frontend memory. They are cleared on the next Start or page reload.
- Durations are estimated between consecutive observations up to 1500 ms apart.
  Longer polling gaps are excluded. The final unpaired observation is not
  extrapolated. View ranges: 5, 15, 30, 60 minutes.
- Export CSV is explicit. Columns: timestamp, activity_state, presence.
  Raw input events, identities, frames, and application names are not exported.
- Appearance choices (Sage/Iris/Copper and Comfortable/Compact) alone are
  saved to localStorage. All other history remains ephemeral.

## Build and validation via kiyoromi (up to 8 Cargo jobs, no local compilation)

Run from local repository root, preserving local uncommitted work:

    git switch feature/live-preview-and-activity-states
    git pull --ff-only
    git status --short
    ssh kiyoromi 'mkdir -p "$HOME/attentracker-build"'
    rsync -az --info=progress2 \
      --exclude='.git/' --exclude='target/' --exclude='node_modules/' \
      --exclude='.env' --exclude='.env.*' --exclude='.venv/' \
      ./ kiyoromi:~/attentracker-build/
    ssh kiyoromi '
      set -eu
      . "$HOME/.cargo/env"
      cd "$HOME/attentracker-build"
      pnpm --dir app/desktop install --frozen-lockfile
      node --test app/desktop/src/activity-history.test.mjs
      pnpm --dir app/desktop build
      cargo fmt --all --check
      CARGO_BUILD_JOBS=8 cargo test --locked --workspace
      CLANG_LIB="$(find /usr/lib -name libclang.so -print -quit)"
      test -n "$CLANG_LIB"
      export LIBCLANG_PATH="$(dirname "$CLANG_LIB")"
      CARGO_BUILD_JOBS=8 cargo test --locked --manifest-path app/desktop/src-tauri/Cargo.toml --bin attentracker-desktop
      CARGO_BUILD_JOBS=8 cargo build --release --locked --offline \
        --features tauri/custom-protocol \
        --manifest-path app/desktop/src-tauri/Cargo.toml
    '
    mkdir -p app/desktop/src-tauri/target/release
    rsync -av --progress \
      kiyoromi:~/attentracker-build/app/desktop/src-tauri/target/release/attentracker-desktop \
      app/desktop/src-tauri/target/release/
    ldd app/desktop/src-tauri/target/release/attentracker-desktop | grep 'not found' || true
    ATTENT_MODEL_PATH="$PWD/models/yunet-dynamic.onnx" \
      ./app/desktop/src-tauri/target/release/attentracker-desktop

If pnpm or a modern Node runtime is absent on the server, provision it
there. The release build --offline flag assumes the server Cargo cache
contains all dependencies; omit it only if cache resolution fails.

## Manual checks

1. Start/Stop preserves the opt-in preview semantics and resets the chart on
   each new session. Frames are not recorded.
2. Change range to verify different windows without restarting the camera.
3. Export creates only categorical timestamps, not an image or raw input log.
4. Clear empties the session chart without affecting inference.
5. Appearance persists after a UI restart; observations do not.

Planned but missing: phone/audio models, expanded activity-state fusion,
opt-in durable local history, trained Kernix classifier, packaged installers,
cross-platform OS activity capture, and external tester validation.
