<script lang="ts">
  import { onMount } from "svelte";
  import { invoke } from "@tauri-apps/api/core";

  type Phase = "stopped" | "starting" | "running" | "stopping" | "error";
  type Presence = "unknown" | "no_face" | "face_present" | "multiple_faces";
  type ActivityState = "unknown" | "working" | "idling" | "away";
  type Snapshot = {
    phase: Phase;
    presence: Presence;
    activityState: ActivityState;
    inputSource: string | null;
    cameraIndex: number | null;
    cameraFormat: string | null;
    framesAnalyzed: number;
    faceCount: number | null;
    bestConfidence: number | null;
    lastObservedAtMs: number | null;
    lastAnalysisMs: number | null;
    lastCaptureMs: number | null;
    lastDecodeMs: number | null;
    lastPreprocessMs: number | null;
    lastInferenceMs: number | null;
    lastPostprocessMs: number | null;
    lastPreviewMs: number | null;
    lastObservationPeriodMs: number | null;
    lastPreviewError: string | null;
    lastError: string | null;
  };
  type FaceOverlay = {
    bbox: [number, number, number, number];
    landmarks: [number, number][];
    confidence: number;
  };
  type FramePreview = {
    sequence: number;
    width: number;
    height: number;
    dataUrl: string;
    faces: FaceOverlay[];
    detectionSequence: number | null;
    detectionAgeMs: number | null;
  };

  const empty: Snapshot = {
    phase: "stopped", presence: "unknown", activityState: "unknown",
    inputSource: null, cameraIndex: null, cameraFormat: null,
    framesAnalyzed: 0, faceCount: null, bestConfidence: null,
    lastObservedAtMs: null, lastAnalysisMs: null, lastCaptureMs: null,
    lastDecodeMs: null, lastPreprocessMs: null, lastInferenceMs: null,
    lastPostprocessMs: null, lastPreviewMs: null, lastObservationPeriodMs: null,
    lastPreviewError: null, lastError: null
  };

  let snapshot = $state<Snapshot>(empty);
  let preview = $state<FramePreview | null>(null);
  let previewEnabled = $state(false);
  let mirror = $state(false);
  let cameraIndex = $state(1);
  let intervalMs = $state(0);
  let modelInputSide = $state(320);
  let commandPending = $state(false);
  let commandError = $state<string | null>(null);
  let lastPreviewSequence = 0;
  let polling = false;
  let previewPolling = false;
  let previewEpoch = 0;
  let lastRenderedSequence = 0;

  // Counters only: no images or raw frame contents enter telemetry.
  function emptyMetrics() {
    return {
      polls: 0, busySkips: 0, pollErrors: 0,
      previewRequests: 0, previewUpdates: 0,
      previewNoChange: 0, previewErrors: 0, renderedImages: 0,
      statusLatencySumMs: 0, statusLatencyMaxMs: 0,
      previewLatencySumMs: 0, previewLatencyMaxMs: 0,
      previewPayloadChars: 0
    };
  }
  let metrics = emptyMetrics();

  async function reportFrontendTelemetry() {
    const report = {
      ...metrics,
      lastReceivedSequence: lastPreviewSequence,
      lastRenderedSequence,
      previewEnabled,
      phase: snapshot.phase
    };
    metrics = emptyMetrics();
    if (report.polls === 0 && report.busySkips === 0 && report.renderedImages === 0) return;
    try {
      await invoke("report_frontend_telemetry", { report });
    } catch (error) {
      console.warn("Failed to report numeric frontend diagnostics", error);
    }
  }

  function previewImageLoaded() {
    if (previewEnabled && preview) {
      lastRenderedSequence = preview.sequence;
      metrics.renderedImages += 1;
    }
  }

  const activityLabels: Record<ActivityState, string> = {
    unknown: "Insufficient evidence",
    working: "Possible working · recent input",
    idling: "Possible idling · no recent input",
    away: "Possibly away · no face for 8 s"
  };

  const states: Record<Presence, string> = {
    unknown: "No observation",
    no_face: "No face detected",
    face_present: "One face detected",
    multiple_faces: "Multiple faces"
  };

  let busy = $derived(
    snapshot.phase === "starting" ||
    snapshot.phase === "running" ||
    snapshot.phase === "stopping"
  );
  let canStart = $derived(
    !busy && !commandPending &&
    Number.isInteger(cameraIndex) && cameraIndex >= 0 &&
    Number.isInteger(intervalMs) && intervalMs >= 0 && intervalMs <= 60_000 &&
    [160, 320, 640].includes(modelInputSide)
  );
  let observedFps = $derived(
    snapshot.lastObservationPeriodMs && snapshot.lastObservationPeriodMs > 0
      ? 1000 / snapshot.lastObservationPeriodMs
      : null
  );
  let unthrottledEstimate = $derived(
    snapshot.lastAnalysisMs && snapshot.lastAnalysisMs > 0
      ? 1000 / snapshot.lastAnalysisMs
      : null
  );

  function fmt(ms: number | null): string {
    return ms == null ? "—" : ms < 10 ? ms.toFixed(1) : ms.toFixed(0);
  }

  let stages = $derived([
    { label: "Camera wait", ms: snapshot.lastCaptureMs },
    { label: "MJPEG decode", ms: snapshot.lastDecodeMs },
    { label: "Tensor preparation", ms: snapshot.lastPreprocessMs },
    { label: "YuNet inference", ms: snapshot.lastInferenceMs },
    { label: "YuNet decode / NMS", ms: snapshot.lastPostprocessMs }
  ]);
  let totalStages = $derived(stages.reduce((sum, part) => sum + (part.ms ?? 0), 0));

  async function refresh() {
    if (polling) {
      metrics.busySkips += 1;
      return;
    }
    polling = true;
    metrics.polls += 1;
    try {
      const statusStarted = performance.now();
      snapshot = await invoke<Snapshot>("monitor_status");
      const statusMs = performance.now() - statusStarted;
      metrics.statusLatencySumMs += statusMs;
      metrics.statusLatencyMaxMs = Math.max(metrics.statusLatencyMaxMs, statusMs);
      if (snapshot.phase === "stopped" || snapshot.phase === "error") {
        // Stopped/Error ends an opt-in camera-preview session.
        if (previewEnabled || preview !== null || lastPreviewSequence !== 0) previewEpoch += 1;
        previewEnabled = false;
        preview = null;
        lastPreviewSequence = 0;
        lastRenderedSequence = 0;
      }
    } catch (error) {
      metrics.pollErrors += 1;
      commandError = String(error);
    } finally {
      polling = false;
    }
  }

  // Preview IPC is independent from status polling. An epoch fence rejects
  // late images after Stop, Hide, or the beginning of another session.
  async function refreshPreview() {
    if (previewPolling || !previewEnabled || snapshot.phase !== "running") return;
    previewPolling = true;
    const epoch = previewEpoch;
    metrics.previewRequests += 1;
    const begun = performance.now();
    try {
      const next = await invoke<FramePreview | null>("monitor_preview", {
        afterSequence: lastPreviewSequence
      });
      const elapsed = performance.now() - begun;
      metrics.previewLatencySumMs += elapsed;
      metrics.previewLatencyMaxMs = Math.max(metrics.previewLatencyMaxMs, elapsed);
      if (epoch !== previewEpoch || !previewEnabled || snapshot.phase !== "running") return;
      if (next && next.sequence > lastPreviewSequence) {
        lastPreviewSequence = next.sequence;
        metrics.previewUpdates += 1;
        metrics.previewPayloadChars += next.dataUrl.length;
        preview = next;
      } else {
        metrics.previewNoChange += 1;
      }
    } catch (error) {
      metrics.previewErrors += 1;
      if (epoch === previewEpoch) commandError = String(error);
    } finally {
      previewPolling = false;
    }
  }

  async function start() {
    previewEpoch += 1;
    commandPending = true;
    commandError = null;
    preview = null;
    lastPreviewSequence = 0;
    lastRenderedSequence = 0;
    try {
      await invoke("start_monitor", {
        cameraIndex: Math.trunc(cameraIndex),
        intervalMs: Math.trunc(intervalMs),
        modelInputSide
      });
      await refresh();
    } catch (e) {
      commandError = String(e);
    } finally {
      commandPending = false;
    }
  }

  async function stop() {
    previewEpoch += 1;
    commandPending = true;
    commandError = null;
    try {
      await invoke("stop_monitor");
      previewEnabled = false;
      preview = null;
      lastPreviewSequence = 0;
      lastRenderedSequence = 0;
      await refresh();
    } catch (e) {
      commandError = String(e);
    } finally {
      commandPending = false;
    }
  }

  async function togglePreview() {
    commandPending = true;
    commandError = null;
    const next = !previewEnabled;
    previewEpoch += 1;
    try {
      await invoke("set_preview_enabled", { enabled: next });
      previewEnabled = next;
      if (!next) {
        preview = null;
        lastPreviewSequence = 0;
        lastRenderedSequence = 0;
      }
    } catch (e) {
      commandError = String(e);
    } finally {
      commandPending = false;
    }
  }

  onMount(() => {
    void refresh();
    const timer = window.setInterval(() => { void refresh(); }, 400);
    const previewTimer = window.setInterval(() => { void refreshPreview(); }, 80);
    const diagnosticsTimer = window.setInterval(() => { void reportFrontendTelemetry(); }, 5000);
    return () => {
      window.clearInterval(timer);
      window.clearInterval(previewTimer);
      window.clearInterval(diagnosticsTimer);
      previewEpoch += 1;
      void reportFrontendTelemetry();
      // Explicitly disable the opt-in image buffer when the window unmounts.
      void invoke("set_preview_enabled", { enabled: false });
    };
  });
</script>

<svelte:head>
  <title>Attentracker — Observation Studio</title>
</svelte:head>

<main>
  <header class="topbar">
    <div class="identity">
      <div class="monogram">a<span>.</span></div>
      <div>
        <div class="brand">attentracker <span class="brand-section">/ studio</span></div>
        <div class="brand-subtitle">Local computer vision · Development build</div>
      </div>
    </div>
    <div class="top-status">
      <span class:active={snapshot.phase === "running"} class:starting={snapshot.phase === "starting"} class="phase">
        <span class="indicator"></span>{snapshot.phase}
      </span>
      <span class="version">YuNet · ONNX · {modelInputSide}px</span>
    </div>
  </header>

  <div class="page-heading">
    <div>
      <p class="eyebrow">01 / Observation</p>
      <h1>Camera diagnostics</h1>
      <p class="subtitle">A smooth opt-in live view with the most recent detections. Overlays may lag behind the camera image.</p>
    </div>
    <span class="privacy-mark">LOCAL ONLY <span aria-hidden="true">●</span></span>
  </div>

  <div class="workspace">
    <section class="panel viewer-panel" aria-label="Detection preview">
      <div class="panel-top">
        <div>
          <p class="eyebrow">Model view</p>
          <h2>Detection preview</h2>
        </div>
        <span class="mini-tag">{preview ? `FRAME #${preview.sequence}` : "PREVIEW OFF"}</span>
      </div>

      <div class="camera-frame" class:mirrored={mirror} style={`aspect-ratio: ${preview ? preview.width / preview.height : 4 / 3}`}>
        {#if preview && previewEnabled}
          <img class="camera-image" src={preview.dataUrl} alt="Live local camera frame with optionally approximate latest detections" onload={previewImageLoaded} />
          <svg
            class="detection-overlay"
            viewBox={`0 0 ${preview.width} ${preview.height}`}
            preserveAspectRatio="none"
            aria-label="Detected face rectangles and landmarks"
          >
            {#each preview.faces as face, index (index)}
              <rect
                class="face-rectangle"
                x={face.bbox[0]} y={face.bbox[1]}
                width={face.bbox[2]} height={face.bbox[3]}
                rx="3"
              />
              <rect
                class="face-label-background"
                x={face.bbox[0]}
                y={Math.max(0, face.bbox[1] - 24)}
                width="100" height="23" rx="3"
              />
              <text class="face-label" x={face.bbox[0] + 7} y={Math.max(15, face.bbox[1] - 8)}>
                FACE {index + 1} · {Math.round(face.confidence * 100)}%
              </text>
              {#each face.landmarks as point, i (i)}
                <circle class="landmark" cx={point[0]} cy={point[1]} r="3.2" />
              {/each}
            {/each}
          </svg>
          <div class="camera-overlay-caption">
            LIVE CAMERA · {preview.detectionSequence == null
              ? "NO RECENT DETECTION OVERLAY"
              : `LATEST ANALYSIS #${preview.detectionSequence} · ${Math.round(preview.detectionAgeMs ?? 0)} MS OLD (APPROXIMATE)`}
          </div>
        {:else}
          <div class="preview-placeholder">
            <div class="camera-symbol" aria-hidden="true">
              <svg viewBox="0 0 64 64" fill="none">
                <rect x="11" y="20" width="33" height="25" rx="5" stroke="currentColor" stroke-width="2"/>
                <path d="M44 27 55 20v25L44 38V27Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
                <path d="M23 16v-6m-4 3h8" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
              </svg>
            </div>
            <p class="placeholder-heading">{previewEnabled ? "Waiting for live camera frames" : "Camera preview is private"}</p>
            <p class="placeholder-detail">
              {previewEnabled ? "Start monitoring or wait for the camera to deliver a frame." : "Enable local live preview to inspect approximate detection overlays. It is disabled by default."}
            </p>
          </div>
        {/if}
      </div>

      <div class="viewer-toolbar">
        <button class:enabled={previewEnabled} class="preview-toggle" onclick={togglePreview} disabled={commandPending || snapshot.phase !== "running"}>
          <span class="toggle-dot"></span>
          {previewEnabled ? "Hide camera preview" : "Show camera preview"}
        </button>
        <label class="mirror-option"><input type="checkbox" bind:checked={mirror} /> Mirror view</label>
        <span class="stream-meta">{snapshot.cameraFormat ?? "No camera selected"}</span>
      </div>
    </section>

    <aside class="panel controls-panel">
      <div class="panel-top">
        <div>
          <p class="eyebrow">Session</p>
          <h2>Controls</h2>
        </div>
      </div>

      <div class="observed-state">
        <div class="state-indicator" class:detected={snapshot.activityState === "working"}></div>
        <div>
          <p class="small-label">Estimated state · provisional</p>
          <p class="state-name">{activityLabels[snapshot.activityState]}</p>
        </div>
      </div>
      <p class="aside-note">Camera: {states[snapshot.presence]}. Input evidence: {snapshot.inputSource ?? "unavailable"}. Activity is not productivity; Working and Idling require an OS input source.</p>

      <div class="config-fields">
        <label>Camera index
          <input type="number" min="0" step="1" bind:value={cameraIndex} disabled={busy || commandPending} />
        </label>
        <label>YuNet model input
          <select bind:value={modelInputSide} disabled={busy || commandPending}>
            <option value={160}>160 × 160 · fastest / lower accuracy</option>
            <option value={320}>320 × 320 · balanced</option>
            <option value={640}>640 × 640 · detailed / slower</option>
          </select>
        </label>
        <label>Target sample period <span>(ms; 0 = max rate)</span>
          <input type="number" min="0" max="60000" step="50" bind:value={intervalMs} disabled={busy || commandPending} />
        </label>
      </div>

      <div class="rate-presets" aria-label="Sampling rate presets">
        <button type="button" class:selected={intervalMs === 1000} onclick={() => intervalMs = 1000} disabled={busy || commandPending}>Eco · 1 FPS</button>
        <button type="button" class:selected={intervalMs === 250} onclick={() => intervalMs = 250} disabled={busy || commandPending}>Balanced · 250 ms</button>
        <button type="button" class:selected={intervalMs === 0} onclick={() => intervalMs = 0} disabled={busy || commandPending}>Max throughput</button>
      </div>
      <p class="rate-hint">Input size changes model cost and detection sensitivity, especially for small or distant faces. Sampling limits apply between passes; 0 ms runs without deliberate sleep. Stop before changing input size.</p>

      <div class="session-actions">
        <button class="start-button" onclick={start} disabled={!canStart}>
          <span class="action-symbol">▶</span> Start monitoring
        </button>
        <button class="stop-button" onclick={stop} disabled={!busy || commandPending || snapshot.phase === "stopping"}>
          <span class="action-symbol">■</span> Stop
        </button>
      </div>

      <div class="control-divider"></div>

      <div class="stat-grid">
        <div class="stat">
          <p class="small-label">Frames analyzed</p>
          <p class="stat-value">{snapshot.framesAnalyzed.toLocaleString()}</p>
        </div>
        <div class="stat">
          <p class="small-label">Effective analyze FPS</p>
          <p class="stat-value">{observedFps == null ? "—" : observedFps.toFixed(2)}<span class="unit"> fps</span></p>
        </div>
        <div class="stat">
          <p class="small-label">Faces in frame</p>
          <p class="stat-value">{snapshot.faceCount ?? "—"}</p>
        </div>
        <div class="stat">
          <p class="small-label">Confidence</p>
          <p class="stat-value">{snapshot.bestConfidence == null ? "—" : (snapshot.bestConfidence * 100).toFixed(1)}<span class="unit">{snapshot.bestConfidence == null ? "" : "%"}</span></p>
        </div>
      </div>

      <p class="aside-note">No monitoring starts until you press Start. No keystrokes or video frames are recorded to disk.</p>
    </aside>
  </div>

  <section class="panel performance-panel" aria-label="Performance profiler">
    <div class="performance-heading">
      <div>
        <p class="eyebrow">02 / Profiler</p>
        <h2>Pipeline latency</h2>
      </div>
      <span class="total-latency">{fmt(snapshot.lastAnalysisMs)} <span>ms / analyzed frame</span></span>
    </div>
    <div class="profile-grid">
      <div class="stage-list">
        {#each stages as stage (stage.label)}
          <div class="stage-row">
            <div class="stage-title">
              <span>{stage.label}</span>
              <strong>{fmt(stage.ms)} ms</strong>
            </div>
            <div class="track"><div class="fill" style={`width: ${totalStages ? Math.max(0, (stage.ms ?? 0) / totalStages * 100) : 0}%`}></div></div>
          </div>
        {/each}
      </div>
      <div class="profile-details">
        <div>
          <p class="small-label">Observed sample period</p>
          <p class="detail-value">{fmt(snapshot.lastObservationPeriodMs)} <span>ms</span></p>
        </div>
        <div>
          <p class="small-label">Async preview JPEG encode</p>
          <p class="detail-value">{fmt(snapshot.lastPreviewMs)} <span>ms</span></p>
        </div>
        <div>
          <p class="small-label">Last observation</p>
          <p class="detail-value time-value">{snapshot.lastObservedAtMs ? new Date(snapshot.lastObservedAtMs).toLocaleTimeString() : "—"}</p>
        </div>
      </div>
    </div>
    <p class="measurement-note">Estimated single-worker max throughput from the latest pass: {unthrottledEstimate == null ? "—" : unthrottledEstimate.toFixed(2)} FPS (not a guarantee). Inference runs sequentially; preview JPEG encoding is asynchronous. A requested sampling period is not a guaranteed FPS. Release builds may be faster than debug builds.</p>
  </section>

  {#if snapshot.lastError}
    <div class="error-panel"><strong>Monitor error</strong><span>{snapshot.lastError}</span></div>
  {/if}
  {#if snapshot.lastPreviewError}
    <div class="error-panel"><strong>Preview encoding error</strong><span>{snapshot.lastPreviewError}</span></div>
  {/if}
  {#if commandError}
    <div class="error-panel"><strong>Application error</strong><span>{commandError}</span></div>
  {/if}

  <footer>
    <span>Attentracker / Observation Studio</span>
    <span>Preview is opt-in · Frames stay in memory · No uploads or recordings</span>
  </footer>
</main>
