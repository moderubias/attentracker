<script lang="ts">
  import { untrack } from "svelte";
  import {
    STATES, summarizeHistory, readableMs, historyCsv,
    type ActivitySample, type TrackedState, type Presence
  } from "./activity-history";

  type Phase = "stopped" | "starting" | "running" | "stopping" | "error";
  type Props = {
    phase: Phase;
    activityState: TrackedState;
    presence: Presence;
    framesAnalyzed: number;
    lastObservedAtMs: number | null;
  };
  let { phase, activityState, presence, framesAnalyzed, lastObservedAtMs }: Props = $props();
  let samples = $state<ActivitySample[]>([]);
  let windowMinutes = $state(15);
  let previousPhase: Phase = "stopped";
  let lastFrameCount = 0;
  let lastObservedTime = 0;
  const stateLabels: Record<TrackedState, string> = {
    working: "Possible working", idling: "Possible idling",
    away: "Possibly away", unknown: "Unknown"
  };

  // Sample the numerical Rust snapshot. No image, key or audio information enters history.
  $effect(() => {
    const currentPhase = phase;
    const sequence = framesAnalyzed;
    const time = lastObservedAtMs;
    const state = activityState;
    const seen = presence;
    if (currentPhase === "running" && previousPhase !== "running") {
      samples = [];
      lastFrameCount = 0;
      lastObservedTime = 0;
    }
    previousPhase = currentPhase;
    if (currentPhase !== "running" || time == null ||
        sequence <= lastFrameCount || time <= lastObservedTime) return;
    lastFrameCount = sequence;
    lastObservedTime = time;
    if (Math.abs(Date.now() - time) > 3000) return;
    const previous = untrack(() => samples);
    samples = [...previous.filter((item) => item.at >= time - 60 * 60000),
      { at: time, state, presence: seen }];
  });

  let summary = $derived(summarizeHistory(samples, windowMinutes, Date.now()));
  let bars = $derived(summary.bins.map((bin, index) => {
    let bottom = 82;
    const segments = STATES.map((state) => {
      const height = Math.min(72, Math.max(0, bin[state] / summary.binMs * 72));
      bottom -= height;
      return { state, y: bottom, height };
    }).filter((segment) => segment.height > 0);
    return { x: index * 10 + 1, segments };
  }));

  function exportCsv() {
    if (!samples.length) return;
    const blob = new Blob([historyCsv(samples)], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "attentracker-session-" + new Date().toISOString().replace(/[:.]/g, "-") + ".csv";
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  }
</script>

<section class="panel insights-panel" aria-label="Session activity timeline">
  <div class="performance-heading insights-header">
    <div>
      <p class="eyebrow">02 / Session insights</p>
      <h2>Observed activity timeline</h2>
      <p class="insights-description">Provisional camera and OS-derived states, not productivity scores.</p>
    </div>
    <div class="insights-actions">
      <label for="history-range">Range</label>
      <select id="history-range" bind:value={windowMinutes}>
        <option value={5}>5 min</option>
        <option value={15}>15 min</option>
        <option value={30}>30 min</option>
        <option value={60}>60 min</option>
      </select>
      <button type="button" onclick={exportCsv} disabled={!samples.length}>Export CSV</button>
      <button type="button" onclick={() => samples = []} disabled={!samples.length}>Clear</button>
    </div>
  </div>
  <div class="insights-stats">
    {#each STATES as state (state)}
      <div class="insights-stat">
        <span class="insights-swatch activity-{state}"></span>
        <div>
          <p class="small-label">{stateLabels[state]}</p>
          <p class="insights-duration">{readableMs(summary.totals[state])}</p>
        </div>
      </div>
    {/each}
  </div>
  <div class="insights-chart">
    {#if samples.length < 2}
      <p class="insights-empty">Start monitoring to collect numerical observations. No history is written to disk.</p>
    {/if}
    <svg viewBox="0 0 600 90" preserveAspectRatio="none" role="img" aria-label="Observed activity states in sixty time buckets">
      <line x1="0" x2="600" y1="82" y2="82" class="insights-axis"></line>
      {#each bars as bar, index (index)}
        {#each bar.segments as segment (segment.state)}
          <rect x={bar.x} y={segment.y} width="8" height={segment.height} rx="1" class="activity-{segment.state}"></rect>
        {/each}
      {/each}
    </svg>
    <div class="insights-ticks"><span>{windowMinutes} minutes ago</span><span>Latest available observations</span></div>
  </div>
  <p class="insights-note">
    {readableMs(summary.observedMs)} of adjacent observations classified in this range.
    Intervals beyond 1.5 s and the final unpaired sample are excluded; missing evidence is not inferred as work.
    The chart resets on the next session; CSV is exported only on request.
  </p>
</section>
