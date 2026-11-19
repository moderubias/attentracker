//! Bounded, aggregation-only telemetry. No camera pixels or frame payloads.
use std::time::{Duration, Instant};
use tracing::info;

const REPORT_INTERVAL: Duration = Duration::from_secs(5);

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FrameTimings {
    pub capture_ms: f64,
    pub decode_ms: f64,
    pub preprocess_ms: f64,
    pub inference_ms: f64,
    pub postprocess_ms: f64,
    pub preview_ms: f64,
    pub pipeline_ms: f64,
    pub iteration_ms: f64,
    pub period_ms: Option<f64>,
    pub planned_sleep_ms: f64,
}

/// Maintains only recent numeric observations for a five-second summary.
pub(crate) struct DiagnosticsWindow {
    started: Instant,
    timings: Vec<FrameTimings>,
}

impl DiagnosticsWindow {
    pub(crate) fn new() -> Self {
        Self {
            started: Instant::now(),
            timings: Vec::new(),
        }
    }

    pub(crate) fn record(&mut self, timings: FrameTimings) {
        self.timings.push(timings);
    }

    pub(crate) fn report_if_due(&mut self, interval_ms: u128, preview_enabled: bool) {
        if self.started.elapsed() >= REPORT_INTERVAL {
            self.report("periodic", interval_ms, preview_enabled);
        }
    }

    pub(crate) fn report(
        &mut self,
        reason: &str,
        interval_ms: u128,
        preview_enabled: bool,
    ) {
        let elapsed = self.started.elapsed().as_secs_f64();
        let frames = self.timings.len();
        if frames == 0 {
            self.started = Instant::now();
            return;
        }

        let fps = frames as f64 / elapsed.max(f64::EPSILON);
        let inference_avg_ms = average(&self.timings, |x| x.inference_ms);
        let inference_p50_ms = percentile(&self.timings, |x| x.inference_ms, 50);
        let inference_p95_ms = percentile(&self.timings, |x| x.inference_ms, 95);
        let inference_max_ms = max_value(&self.timings, |x| x.inference_ms);
        let pipeline_avg_ms = average(&self.timings, |x| x.pipeline_ms);
        let iteration_avg_ms = average(&self.timings, |x| x.iteration_ms);
        let iteration_p95_ms = percentile(&self.timings, |x| x.iteration_ms, 95);
        let period_avg_ms = average_present(&self.timings, |x| x.period_ms);

        info!(
            target: "attent_core::telemetry",
            event = "monitor_window",
            reason,
            frames,
            elapsed_s = elapsed,
            fps,
            target_interval_ms = interval_ms,
            preview_enabled,
            capture_avg_ms = average(&self.timings, |x| x.capture_ms),
            decode_avg_ms = average(&self.timings, |x| x.decode_ms),
            preprocess_avg_ms = average(&self.timings, |x| x.preprocess_ms),
            inference_avg_ms,
            inference_p50_ms,
            inference_p95_ms,
            inference_max_ms,
            postprocess_avg_ms = average(&self.timings, |x| x.postprocess_ms),
            preview_avg_ms = average(&self.timings, |x| x.preview_ms),
            pipeline_avg_ms,
            iteration_avg_ms,
            iteration_p95_ms,
            period_avg_ms,
            planned_sleep_avg_ms = average(&self.timings, |x| x.planned_sleep_ms),
            "monitor performance summary"
        );

        self.timings.clear();
        self.started = Instant::now();
    }
}

fn average(values: &[FrameTimings], f: impl Fn(&FrameTimings) -> f64) -> f64 {
    values.iter().map(f).sum::<f64>() / values.len() as f64
}

fn max_value(values: &[FrameTimings], f: impl Fn(&FrameTimings) -> f64) -> f64 {
    values.iter().map(f).fold(0.0_f64, f64::max)
}

fn average_present(values: &[FrameTimings], f: impl Fn(&FrameTimings) -> Option<f64>) -> f64 {
    let (sum, count) = values
        .iter()
        .filter_map(f)
        .fold((0.0_f64, 0_usize), |(sum, count), v| (sum + v, count + 1));
    if count == 0 { 0.0 } else { sum / count as f64 }
}

fn percentile(
    values: &[FrameTimings],
    f: impl Fn(&FrameTimings) -> f64,
    percent: usize,
) -> f64 {
    let mut sorted: Vec<f64> = values.iter().map(f).collect();
    sorted.sort_unstable_by(f64::total_cmp);
    let index = (sorted.len() * percent).div_ceil(100).saturating_sub(1);
    sorted[index.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_uses_nearest_rank() {
        let values: Vec<_> = (1..=20)
            .rev()
            .map(|i| FrameTimings {
                inference_ms: i as f64,
                ..FrameTimings::default()
            })
            .collect();
        assert_eq!(percentile(&values, |x| x.inference_ms, 50), 10.0);
        assert_eq!(percentile(&values, |x| x.inference_ms, 95), 19.0);
        assert_eq!(max_value(&values, |x| x.inference_ms), 20.0);
        assert!((average(&values, |x| x.inference_ms) - 10.5).abs() < 1e-10);
    }

    #[test]
    fn missing_period_does_not_distort_average() {
        let values = [
            FrameTimings { period_ms: None, ..FrameTimings::default() },
            FrameTimings { period_ms: Some(250.0), ..FrameTimings::default() },
            FrameTimings { period_ms: Some(500.0), ..FrameTimings::default() },
        ];
        assert_eq!(average_present(&values, |x| x.period_ms), 375.0);
    }
}
