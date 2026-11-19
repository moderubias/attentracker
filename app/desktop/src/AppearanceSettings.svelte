<script lang="ts">
  import { onMount } from "svelte";
  type Accent = "sage" | "iris" | "copper";
  type Density = "comfortable" | "compact";
  const KEY = "attentracker.appearance.v1";
  let accent = $state<Accent>("sage");
  let density = $state<Density>("comfortable");
  function apply() {
    document.documentElement.dataset.accent = accent;
    document.documentElement.dataset.density = density;
  }
  function save() {
    apply();
    try { localStorage.setItem(KEY, JSON.stringify({ accent, density })); } catch { /* storage optional */ }
  }
  function selectAccent(value: Accent) { accent = value; save(); }
  function selectDensity(value: Density) { density = value; save(); }
  onMount(() => {
    try {
      const stored = localStorage.getItem(KEY);
      if (stored) {
        const config = JSON.parse(stored) as Partial<{ accent: Accent; density: Density }>;
        if (config.accent === "sage" || config.accent === "iris" || config.accent === "copper") accent = config.accent;
        if (config.density === "comfortable" || config.density === "compact") density = config.density;
      }
    } catch { /* malformed or blocked settings */ }
    apply();
  });
</script>

<section class="panel appearance-panel" aria-label="Workspace appearance">
  <div>
    <p class="eyebrow">03 / Personalize</p>
    <h2>Workspace appearance</h2>
    <p class="insights-description">Only appearance preferences are retained locally, never camera observations.</p>
  </div>
  <div class="appearance-controls">
    <div class="appearance-group">
      <span class="appearance-caption">Accent palette</span>
      <div class="appearance-options">
        <button type="button" class:chosen={accent === "sage"} aria-pressed={accent === "sage"} onclick={() => selectAccent("sage")}><span class="swatch sage"></span>Sage</button>
        <button type="button" class:chosen={accent === "iris"} aria-pressed={accent === "iris"} onclick={() => selectAccent("iris")}><span class="swatch iris"></span>Iris</button>
        <button type="button" class:chosen={accent === "copper"} aria-pressed={accent === "copper"} onclick={() => selectAccent("copper")}><span class="swatch copper"></span>Copper</button>
      </div>
    </div>
    <div class="appearance-group">
      <span class="appearance-caption">Layout density</span>
      <div class="appearance-options">
        <button type="button" class:chosen={density === "comfortable"} aria-pressed={density === "comfortable"} onclick={() => selectDensity("comfortable")}>Comfortable</button>
        <button type="button" class:chosen={density === "compact"} aria-pressed={density === "compact"} onclick={() => selectDensity("compact")}>Compact</button>
      </div>
    </div>
  </div>
</section>
