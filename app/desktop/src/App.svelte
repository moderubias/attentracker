<script lang="ts">
  import { onMount } from "svelte";
  import { invoke } from "@tauri-apps/api/core";

  type Phase = "stopped" | "starting" | "running" | "stopping" | "error";
  type Presence = "unknown" | "no_face" | "face_present" | "multiple_faces";

  type Snapshot = {
    phase: Phase;
    presence: Presence;
    cameraIndex: number | null;
    framesAnalyzed: number;
    faceCount: number | null;
    bestConfidence: number | null;
    lastObservedAtMs: number | null;
    lastAnalysisMs: number | null;
    lastCaptureMs: number | null;
    lastError: string | null;
  };

  const empty: Snapshot = {
    phase: "stopped",
    presence: "unknown",
    cameraIndex: null,
    framesAnalyzed: 0,
    faceCount: null,
    bestConfidence: null,
    lastObservedAtMs: null,
    lastAnalysisMs: null,
    lastCaptureMs: null,
    lastError: null
  };

  let snapshot = $state<Snapshot>(empty);
  let cameraIndex = $state(1);
  let intervalMs = $state(3000);
  let pending = $state(false);
  let commandError = $state<string | null>(null);

  const readablePresence: Record<Presence, string> = {
    unknown: "Unknown",
    no_face: "No face",
    face_present: "Face detected",
    multiple_faces: "Multiple faces"
  };

  async function refresh() {
    try {
      snapshot = await invoke<Snapshot>("monitor_status");
    } catch (e) {
      commandError = String(e);
    }
  }

  async function start() {
    pending = true;
    commandError = null;
    try {
      await invoke("start_monitor", {
        cameraIndex: Math.trunc(cameraIndex),
        intervalMs: Math.trunc(intervalMs)
      });
      await refresh();
    } catch (e) {
      commandError = String(e);
    } finally {
      pending = false;
    }
  }

  async function stop() {
    pending = true;
    commandError = null;
    try {
      await invoke("stop_monitor");
      await refresh();
    } catch (e) {
      commandError = String(e);
    } finally {
      pending = false;
    }
  }

  onMount(() => {
    void refresh();
    const timer = window.setInterval(() => { void refresh(); }, 750);
    return () => window.clearInterval(timer);
  });

  let busy = $derived(snapshot.phase === "running" || snapshot.phase === "starting" || snapshot.phase === "stopping");
</script>

<svelte:head>
  <title>Attentracker — Monitor</title>
</svelte:head>

<main>
  <header>
    <div>
      <p class="eyebrow">Local observation · Early prototype</p>
      <h1>Attentracker<span style="color:#91a99a">.</span></h1>
      <p class="subhead">Private, on-device computer vision monitoring.</p>
    </div>
    <span class="pill">{snapshot.phase}</span>
  </header>

  <div class="cards">
    <section class="card">
      <p class="eyebrow">Camera observation</p>
      <div class="metric small">{readablePresence[snapshot.presence]}</div>
      <p class="muted">No claim about productivity or identity</p>
    </section>
    <section class="card">
      <p class="eyebrow">Frames analyzed</p>
      <div class="metric">{snapshot.framesAnalyzed}</div>
      <p class="muted">{snapshot.faceCount ?? "—"} face(s) in latest observation</p>
    </section>
    <section class="card">
      <p class="eyebrow">Analysis latency</p>
      <div class="metric">{snapshot.lastAnalysisMs == null ? "—" : snapshot.lastAnalysisMs.toFixed(0)}<span style="font-size:16px"> ms</span></div>
      <p class="muted">Capture + preprocess + inference + decode</p>
    </section>
  </div>

  <section class="card">
    <h2>Monitor controls</h2>
    <p class="subhead">The camera is accessed only after pressing Start. No frames are sent to the frontend or written to disk.</p>
    <div class="actions">
      <label class="field">Camera index
        <input type="number" min="0" step="1" bind:value={cameraIndex} disabled={busy || pending} />
      </label>
      <label class="field">Sample interval (ms)
        <input type="number" min="250" max="60000" step="250" bind:value={intervalMs} disabled={busy || pending} />
      </label>
      <button class="primary" onclick={start} disabled={busy || pending || !Number.isInteger(cameraIndex) || cameraIndex < 0 || !Number.isInteger(intervalMs) || intervalMs < 250 || intervalMs > 60000}>Start monitoring</button>
      <button onclick={stop} disabled={!busy || pending || snapshot.phase === "stopping"}>Stop</button>
    </div>
    {#if snapshot.lastError}
      <div class="error">Runtime: {snapshot.lastError}</div>
    {/if}
    {#if commandError}
      <div class="error">Command: {commandError}</div>
    {/if}
    <div class="notice">
      <strong>Not a productivity score.</strong> YuNet reports visible faces only. Working, idle, away, talking, and phone use are not inferred yet. Camera permission and user consent are required. Closing the app stops its monitoring process.
    </div>
  </section>
</main>
