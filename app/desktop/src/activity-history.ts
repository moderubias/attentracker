export type TrackedState = "working" | "idling" | "away" | "unknown";
export type Presence = "unknown" | "no_face" | "face_present" | "multiple_faces";
export interface ActivitySample { at: number; state: TrackedState; presence: Presence }
export const STATES = ["working", "idling", "away", "unknown"] as const;
export const MAX_GAP_MS = 1500;
export type Durations = Record<TrackedState, number>;
export interface Summary { bins: Durations[]; totals: Durations; observedMs: number; binMs: number }
const empty = (): Durations => ({ working: 0, idling: 0, away: 0, unknown: 0 });

/** Only adjacent, fresh observations contribute time; no extrapolation of the last sample. */
export function summarizeHistory(samples: readonly ActivitySample[], minutes: number, now: number): Summary {
  const span = (Number.isFinite(minutes) ? Math.max(1, Math.min(60, Math.floor(minutes))) : 15) * 60000;
  const start = now - span;
  const binMs = span / 60;
  const bins = Array.from({ length: 60 }, empty);
  const totals = empty();
  for (let i = 0; i + 1 < samples.length; i++) {
    const a = samples[i], b = samples[i + 1];
    if (!Number.isFinite(a.at) || !Number.isFinite(b.at) || !STATES.includes(a.state)) continue;
    let cursor = Math.max(start, a.at);
    const end = Math.min(b.at, a.at + MAX_GAP_MS, now);
    if (end <= cursor) continue;
    totals[a.state] += end - cursor;
    while (cursor < end) {
      const index = Math.max(0, Math.min(59, Math.floor((cursor - start) / binMs)));
      const boundary = Math.min(end, start + (index + 1) * binMs);
      bins[index][a.state] += boundary - cursor;
      cursor = boundary;
    }
  }
  return { bins, totals, binMs, observedMs: STATES.reduce((sum, state) => sum + totals[state], 0) };
}

export function readableMs(ms: number): string {
  const seconds = Math.max(0, Math.round(ms / 1000));
  const minutes = Math.floor(seconds / 60);
  return minutes ? String(minutes) + "m " + String(seconds % 60) + "s" : String(seconds) + "s";
}

export function historyCsv(samples: readonly ActivitySample[]): string {
  const lines = ["timestamp,activity_state,presence"];
  for (const sample of samples) {
    if (!Number.isFinite(sample.at) || !STATES.includes(sample.state)) continue;
    lines.push(new Date(sample.at).toISOString() + "," + sample.state + "," + sample.presence);
  }
  return lines.join("\r\n") + "\r\n";
}
