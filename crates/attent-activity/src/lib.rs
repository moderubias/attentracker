//! Conservative temporal interpretation of observable evidence.
//!
//! A face is not proof of identity or productivity. `Working` is a provisional
//! correlation of one consistently visible face and OS input recency.
//! Face detection can miss individual frames, so changes are debounced.

use std::time::{Duration, Instant};
use serde::Serialize;

const AWAY_AFTER: Duration = Duration::from_secs(8);
const FACE_LOSS_GRACE: Duration = Duration::from_millis(750);
const FACE_CONFIRM: Duration = Duration::from_millis(400);
const IDLING_AFTER: Duration = Duration::from_secs(60);
// A 10–15 s idle duration must not be described as "recent input".
// Wayland idle notifications fire after 1 second. An already-idle seat
// (>=1 s) must never qualify as "recent input".
const RECENT_INPUT: Duration = Duration::from_millis(900);
// Enter Working only on fresh evidence, but keep it across short pauses.
// This prevents one-second Wayland idle notifications from fragmenting a
// continuous input session into dozens of artificial state transitions.
const WORKING_EXIT_IDLE: Duration = Duration::from_secs(5);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivityState {
    #[default]
    Unknown,
    ActiveWorking,
    Working,
    MultipleFacesPresent,
    Idling,
    Away,
}

/// Explicitly sourced, approximate OS-wide idle duration when available.
/// None must not be interpreted as input activity or input inactivity.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InputEvidence {
    pub idle_for: Option<Duration>,
}

#[derive(Default)]
pub struct ActivityClassifier {
    absent_since: Option<Instant>,
    present_since: Option<Instant>,
    state: ActivityState,
}

impl ActivityClassifier {
    pub fn new() -> Self { Self::default() }

    /// `face_count=None` means invalid/unavailable camera evidence.
    /// A face present is not necessarily the owner of the computer.
    pub fn observe(
        &mut self, now: Instant, face_count: Option<usize>, input: InputEvidence,
    ) -> ActivityState {
        let next = match face_count {
            None => {
                self.absent_since = None;
                self.present_since = None;
                ActivityState::Unknown
            }
            Some(0) => {
                let absent_since = *self.absent_since.get_or_insert(now);
                let elapsed = now.saturating_duration_since(absent_since);
                if elapsed >= AWAY_AFTER {
                    self.present_since = None;
                    ActivityState::Away
                } else if elapsed < FACE_LOSS_GRACE
                    && self.state != ActivityState::MultipleFacesPresent
                {
                    // Brief single-face misses do not cause spurious state changes.
                    self.state
                } else {
                    self.present_since = None;
                    ActivityState::Unknown
                }
            }
            Some(1) => {
                let present_since = *self.present_since.get_or_insert(now);
                if now.saturating_duration_since(present_since) < FACE_CONFIRM {
                    // Preserve Away until a face has been present long enough;
                    // one false positive must not reset the absence timer.
                    if self.state == ActivityState::Away {
                        ActivityState::Away
                    } else {
                        ActivityState::Unknown
                    }
                } else {
                    self.absent_since = None;
                    match input.idle_for {
                        Some(age) if age >= IDLING_AFTER => ActivityState::Idling,
                        // "Active" means fresh OS input evidence, not verified productivity.
                        Some(age) if age <= RECENT_INPUT => ActivityState::ActiveWorking,
                        // A short pause after fresh input is still a provisional Working state.
                        Some(age)
                            if matches!(self.state, ActivityState::ActiveWorking | ActivityState::Working)
                                && age < WORKING_EXIT_IDLE =>
                        {
                            ActivityState::Working
                        }
                        _ => ActivityState::Unknown,
                    }
                }
            }
            Some(_) => {
                // Count is observed, but identity and intent remain unknown.
                // Previous single-face input confidence must not carry over.
                self.absent_since = None;
                self.present_since = None;
                ActivityState::MultipleFacesPresent
            }
        };
        self.state = next;
        next
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idle_s(seconds: u64) -> InputEvidence {
        InputEvidence { idle_for: Some(Duration::from_secs(seconds)) }
    }
    fn idle_ms(ms: u64) -> InputEvidence {
        InputEvidence { idle_for: Some(Duration::from_millis(ms)) }
    }

    #[test]
    fn away_requires_eight_seconds_without_face() {
        let at = Instant::now();
        let mut c = ActivityClassifier::new();
        let missing = InputEvidence::default();
        assert_eq!(c.observe(at, Some(0), missing), ActivityState::Unknown);
        assert_eq!(c.observe(at + Duration::from_secs(7), Some(0), missing), ActivityState::Unknown);
        assert_eq!(c.observe(at + Duration::from_secs(8), Some(0), missing), ActivityState::Away);
    }

    #[test]
    fn active_working_requires_confirmed_face_and_fresh_input() {
        let at = Instant::now();
        let mut c = ActivityClassifier::new();
        assert_eq!(c.observe(at, Some(1), idle_ms(0)), ActivityState::Unknown);
        assert_eq!(c.observe(at + Duration::from_millis(500), Some(1), idle_ms(900)), ActivityState::ActiveWorking);
        assert_eq!(c.observe(at + Duration::from_millis(600), Some(1), idle_ms(901)), ActivityState::Working);
        assert_eq!(c.observe(at + Duration::from_millis(650), Some(1), idle_s(5)), ActivityState::Unknown);
        assert_eq!(c.observe(at + Duration::from_millis(700), Some(1), idle_s(1)), ActivityState::Unknown);
        assert_eq!(c.observe(at + Duration::from_millis(800), Some(1), idle_ms(0)), ActivityState::ActiveWorking);
    }

    #[test]
    fn brief_missed_face_preserves_state_but_prolonged_loss_does_not() {
        let at = Instant::now();
        let mut c = ActivityClassifier::new();
        c.observe(at, Some(1), idle_s(0));
        assert_eq!(c.observe(at + Duration::from_millis(500), Some(1), idle_s(0)), ActivityState::ActiveWorking);
        assert_eq!(c.observe(at + Duration::from_millis(550), Some(0), idle_s(0)), ActivityState::ActiveWorking);
        assert_eq!(c.observe(at + Duration::from_millis(650), Some(1), idle_s(0)), ActivityState::ActiveWorking);
        assert_eq!(c.observe(at + Duration::from_secs(1), Some(0), idle_s(1)), ActivityState::ActiveWorking);
        assert_eq!(c.observe(at + Duration::from_secs(2), Some(0), idle_s(2)), ActivityState::Unknown);
        assert_eq!(c.observe(at + Duration::from_secs(9), Some(0), idle_s(9)), ActivityState::Away);
    }

    #[test]
    fn brief_false_face_does_not_cancel_away() {
        let at = Instant::now();
        let mut c = ActivityClassifier::new();
        let missing = InputEvidence::default();
        c.observe(at, Some(0), missing);
        assert_eq!(c.observe(at + Duration::from_secs(8), Some(0), missing), ActivityState::Away);
        assert_eq!(c.observe(at + Duration::from_secs(9), Some(1), idle_s(0)), ActivityState::Away);
        assert_eq!(c.observe(at + Duration::from_secs(9) + Duration::from_millis(100), Some(0), idle_s(0)), ActivityState::Away);
        assert_eq!(c.observe(at + Duration::from_secs(10), Some(1), idle_s(0)), ActivityState::Away);
        assert_eq!(c.observe(at + Duration::from_secs(10) + Duration::from_millis(500), Some(1), idle_s(0)), ActivityState::ActiveWorking);
    }

    #[test]
    fn multiple_faces_is_observation_not_work_inference() {
        let at = Instant::now();
        let mut c = ActivityClassifier::new();
        c.observe(at, Some(1), idle_s(0));
        c.observe(at + Duration::from_millis(500), Some(1), idle_s(0));
        assert_eq!(c.observe(at + Duration::from_millis(600), Some(2), idle_s(0)), ActivityState::MultipleFacesPresent);
        assert_eq!(c.observe(at + Duration::from_millis(700), Some(0), idle_s(0)), ActivityState::Unknown);
        assert_eq!(c.observe(at + Duration::from_millis(800), Some(1), idle_s(0)), ActivityState::Unknown);
        assert_eq!(c.observe(at + Duration::from_millis(1300), Some(1), idle_s(0)), ActivityState::ActiveWorking);
        assert_eq!(c.observe(at + Duration::from_millis(1400), Some(3), InputEvidence::default()), ActivityState::MultipleFacesPresent);
    }

    #[test]
    fn returning_after_eleven_seconds_idle_is_not_working() {
        let at = Instant::now();
        let mut c = ActivityClassifier::new();
        c.observe(at, Some(0), InputEvidence::default());
        c.observe(at + Duration::from_secs(8), Some(0), idle_s(10));
        assert_eq!(c.observe(at + Duration::from_secs(9), Some(1), idle_s(11)), ActivityState::Away);
        assert_eq!(c.observe(at + Duration::from_secs(9) + Duration::from_millis(500), Some(1), idle_s(11)), ActivityState::Unknown);
    }

    #[test]
    fn working_hysteresis_does_not_invent_recent_input() {
        let at = Instant::now();
        let mut c = ActivityClassifier::new();
        c.observe(at, Some(1), idle_s(0));
        assert_eq!(c.observe(at + Duration::from_millis(500), Some(1), idle_s(0)), ActivityState::ActiveWorking);
        assert_eq!(c.observe(at + Duration::from_secs(2), Some(1), idle_s(1)), ActivityState::Working);
        assert_eq!(c.observe(at + Duration::from_secs(3), Some(1), idle_s(4)), ActivityState::Working);
        assert_eq!(c.observe(at + Duration::from_secs(4), Some(1), idle_s(5)), ActivityState::Unknown);
        assert_eq!(c.observe(at + Duration::from_secs(5), Some(1), idle_s(4)), ActivityState::Unknown);
        assert_eq!(c.observe(at + Duration::from_secs(6), Some(1), idle_s(0)), ActivityState::ActiveWorking);
        assert_eq!(c.observe(at + Duration::from_secs(7), Some(1), InputEvidence::default()), ActivityState::Unknown);
        assert_eq!(c.observe(at + Duration::from_secs(68), Some(1), idle_s(60)), ActivityState::Idling);
    }

    #[test]
    fn input_states_require_os_evidence_and_camera_observation() {
        let at = Instant::now();
        let mut c = ActivityClassifier::new();
        c.observe(at, Some(1), InputEvidence::default());
        assert_eq!(c.observe(at + Duration::from_millis(500), Some(1), InputEvidence::default()), ActivityState::Unknown);
        assert_eq!(c.observe(at + Duration::from_millis(600), Some(1), idle_s(0)), ActivityState::ActiveWorking);
        assert_eq!(c.observe(at + Duration::from_millis(700), Some(1), idle_s(1)), ActivityState::Working);
        assert_eq!(c.observe(at + Duration::from_millis(750), Some(1), idle_s(10)), ActivityState::Unknown);
        assert_eq!(c.observe(at + Duration::from_millis(800), Some(1), idle_s(60)), ActivityState::Idling);
        assert_eq!(c.observe(at + Duration::from_millis(900), Some(2), idle_s(0)), ActivityState::MultipleFacesPresent);
        assert_eq!(c.observe(at + Duration::from_millis(1000), None, idle_s(0)), ActivityState::Unknown);
    }
}

#[cfg(target_os = "linux")]
mod wayland_idle;
#[cfg(target_os = "linux")]
pub use wayland_idle::InputIdleMonitor;

#[cfg(not(target_os = "linux"))]
pub struct InputIdleMonitor;
#[cfg(not(target_os = "linux"))]
impl InputIdleMonitor {
    pub fn start() -> Self { Self }
    pub fn idle_for(&self) -> Option<Duration> { None }
    pub fn source(&self) -> Option<&'static str> { None }
}
