# Live preview and activity-state validation (experimental)

This branch (`feature/live-preview-and-activity-states`) is stacked on the
YuNet optimization PR. The 20–30 analyzed-FPS baseline was measured **before**
decoupling camera capture; do not assume equivalent FPS until benchmarked.

## Architecture

- A single camera owner continuously obtains and decodes frames at its
  negotiated rate, retaining only the newest `Arc<CapturedFrame>`.
- The model worker selects the newest **unprocessed** sequence number; old
  frames are intentionally skipped rather than queued.
- A separate bounded, one-slot JPEG encoder receives opt-in samples from the
  camera thread no more than every 85 ms (roughly 12 preview samples/s).
- The UI polls the latest preview every 80 ms, separately from the 400 ms
  status polling. The rendered rate will be lower if the camera, JPEG encoder
  or webview cannot keep up.
- The preview overlays are taken from the most recent analyzed frame, with a
  **350 ms freshness limit**. These are approximate overlays on a newer
  camera frame; the source detection sequence and age are shown in the UI.
- Turning preview off drops its retained encoded image and prevents an
  in-flight encoder result from being published by using a generation fence.
  It does **not** stop monitoring or bypass the camera input required by ML.
- Stop and camera errors now **revoke preview opt-in**, rather than keeping
  the toggle armed for the next Start. A new session requires an explicit
  Show camera preview action after monitoring is running.
- No image or raw key material is written to disk or sent to a remote server.

## State semantics

| State | Evidence | Interpretation |
| --- | --- | --- |
| `Unknown` | Missing/ambiguous signals; multiple faces; no recent OS input evidence | Insufficient evidence |
| `Away` | No face detected continuously for >=8 seconds | Possibly away, not proof of absence |
| `Working` | One face and OS input observed within the recent-activity threshold | Possible computer activity, not productivity |
| `Idling` | One face and OS input idle for >=60 seconds | Possible computer idling |

The activity classifier applies separate entry/exit thresholds: `Working`
needs a confirmed face plus input-idle evidence below **0.9 seconds** to
start; once active it persists through brief input gaps until idle reaches
**5 seconds**. That is intentionally asymmetric to suppress flicker caused
by Wayland's 1-second `Idled` notifications. The classifier still moves
to `Unknown` immediately if OS input evidence becomes unavailable, and
to `Idling` at 60 seconds with a confirmed face. A brief loss of face
evidence is separately tolerated for 750 ms.

The `Working` state requires an **actual OS input-idle signal**. On Linux
Wayland, the worker attempts to connect to
`ext-idle-notify-v1` **version 2** and use
`get_input_idle_notification`; this reports only `idled` and `resumed`
transitions and ignores idle inhibitors. It never sees keystrokes, window
titles, pointers or key content. A 1 s idle event allows an approximate
input-idle duration. Monitoring starts after pressing **Start** and the
observer is dropped on Stop, Error or process exit.

If the compositor does not support version 2, or under an unsupported OS,
the input evidence is unavailable; the UI must report this truthfully.
`Working` and `Idling` will not be inferred. The face-only `Away`
heuristic can still operate. The camera does not identify *which* person
is visible, and the system must not claim such identification.

State changes are logged as `event="activity_transition"` with only
previous/current provisional states, face count, provider name and the
elapsed idle duration (if available). No key identities, mouse coordinates,
window titles or pixels appear in those diagnostic events. The event
diagnoses transitions; it does not validate whether an inference about
actual work or physical absence is correct.

## Offloaded build (no Rust compiler on the laptop)

From the laptop's repository root:

```sh
cd ~/All_Work_Folders/moderubias/projects/attentracker
git status --short
git fetch origin
git switch --track origin/feature/live-preview-and-activity-states
pnpm --dir app/desktop install --frozen-lockfile
pnpm --dir app/desktop build
rsync -az --info=progress2 \
  --exclude='.git/' --exclude='target/' --exclude='node_modules/' \
  --exclude='.env' --exclude='.env.*' --exclude='.venv/' \
  ./ kiyoromi:~/attentracker-build/
```

If the branch already exists locally, use
`git switch feature/live-preview-and-activity-states` then
`git pull --ff-only`. Preserve uncommitted changes before switching
(`git stash push -u -m "before-live-preview"`); never discard them.

On the Debian build server (over SSH), compile the production Tauri
protocol while reusing its existing `target` directory. **Do not use
`--offline` the first time**: Wayland dependencies may be new.

```sh
ssh kiyoromi '
  set -eu
  . "$HOME/.cargo/env"
  cd "$HOME/attentracker-build"
  CLANG_LIB="$(find /usr/lib -name libclang.so -print -quit)"
  test -n "$CLANG_LIB"
  export LIBCLANG_PATH="$(dirname "$CLANG_LIB")"
  CARGO_BUILD_JOBS=4 cargo build --release \
    --features tauri/custom-protocol \
    --manifest-path app/desktop/src-tauri/Cargo.toml
'
```

Back on the laptop:

```sh
mkdir -p app/desktop/src-tauri/target/release
rsync -av --progress \
  kiyoromi:~/attentracker-build/app/desktop/src-tauri/target/release/attentracker-desktop \
  app/desktop/src-tauri/target/release/
ATTENT_MODEL_PATH="$PWD/models/yunet-dynamic.onnx" \
RUST_LOG='warn,attentracker_desktop=info,attent_core=info,attent_camera=info,attent_model=info,attent_activity=info' \
  ./app/desktop/src-tauri/target/release/attentracker-desktop \
  2>&1 | tee "$HOME/attentracker-live-preview.log"
```

The `models/yunet-dynamic.onnx` file must already be present and
checksum-verified with `bash scripts/fetch-yunet-dynamic.sh`. The
app still uses the original fixed model at 640 when it is available.

## On-device acceptance checks

1. Choose **320×320**, Max, preview **off**, and observe analyzed FPS
   for >=30 s. Compare against the previous 20–21 FPS result.
2. Enable preview for >=30 s and confirm responsive live video with
   overlays that show their age and source sequence. Face boxes are
   approximations, not model-synchronized overlays.
3. Check `event="capture_window"`, `event="preview_window"`,
   `event="frontend_window"` and `event="monitor_window"`.
   Capture FPS, encoded-preview FPS, frontend rendered-image count and
   analyzed FPS are **different measurements**.
4. Hide preview: camera image disappears immediately. Stop monitoring:
   no more capture/idle readings are made after worker shutdown.
5. Remain out of frame for >=8 s; observe `Away`. Re-enter the frame
   and see `Unknown` or input-derived `Working/Idling`.
6. On Niri, if `input_source` becomes Wayland, test typing/mouse
   activity. Continuous user activity with ~1-second gaps should not
   alternate `Working`/`Unknown` on every idle notification.
   Then leave input untouched for >=60 s while visible.
   The UI should show provisional `Working` and `Idling`
   appropriately. Do not use automated synthetic input.
7. Toggle Stop/Start repeatedly and verify the app opens the camera
   exactly once per session and no preview frames from an old session
   reappear.

```sh
python scripts/throughput-report.py "$HOME/attentracker-live-preview.log"
rg 'capture_window|preview_window|frontend_window|monitor_window|ERROR|WARN|input idle' \
  "$HOME/attentracker-live-preview.log" | tail -n 150
```

If the Wayland provider does not connect, record its `attent_activity::idle`
diagnostic message, `echo "$XDG_SESSION_TYPE"`, and
`echo "${WAYLAND_DISPLAY:-unset}"`; do not enable global keyboard
hooks or require elevated access to `/dev/input` as a workaround.
