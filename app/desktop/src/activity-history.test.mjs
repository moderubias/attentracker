import { test } from "node:test";
import assert from "node:assert/strict";
import { summarizeHistory, historyCsv } from "./activity-history.ts";
const sample = (at, state = "working") => ({ at, state, presence: "face_present" });
test("long gaps do not invent working time", () => {
  const result = summarizeHistory([sample(1000), sample(4000)], 5, 5000);
  assert.equal(result.totals.working, 1500);
  assert.equal(result.observedMs, 1500);
});
test("last sample never accumulates time", () => {
  assert.equal(summarizeHistory([sample(1000)], 5, 5000).observedMs, 0);
});
test("window clipping and bins match totals", () => {
  const result = summarizeHistory([sample(1000), sample(2000, "idling"), sample(3000)], 1, 61500);
  assert.equal(result.totals.working, 500);
  assert.equal(result.totals.idling, 1000);
  assert.equal(result.bins.reduce((sum, bin) => sum + bin.working, 0), 500);
});
test("separate states and explicit CSV", () => {
  const samples = [sample(100, "away"), sample(600, "unknown"), sample(900)];
  const result = summarizeHistory(samples, 1, 1000);
  assert.equal(result.totals.away, 500);
  assert.equal(result.totals.unknown, 300);
  const csv = historyCsv(samples);
  assert.match(csv, /^timestamp,activity_state,presence\r\n/);
  assert.match(csv, /,away,face_present/);
});
