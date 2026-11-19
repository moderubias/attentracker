//! Conservative temporal interpretation of observable evidence.
//!
//! A face is not proof of identity or productivity. `Working` is a provisional
//! correlation of one consistently visible face and recent OS input activity.
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
    Working,
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
                } else if elapsed < FACE_LOSS_GRACE {
                    // Keep previous face confirmation through a brief miss.
                    // This prevents Working -> Unknown -> Working flicker.
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
                        Some(age) if age <= RECENT_INPUT => ActivityState::Working,
                        Some(age)
                            if self.state == ActivityState::Working && age < WORKING_EXIT_IDLE =>
                        {
                            ActivityState::Working
                        }
                        _ => ActivityState::Unknown,
                    }
                }
            }
            Some(_) => {
                // Multiple people are ambiguous; previous single-face
                // confidence does not carry across this observation.
                self.absent_since = None;
                self.present_since = None;
                ActivityState::Unknown
            }
        };
        self.state = next;
        next
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idle(seconds: u64) -> InputEvidence {
        InputEvidence { idle_for: Some(Duration::from_secs(seconds)) }
    }

    #[test]
    fn away_requires_eight_seconds_without_face() {
        let at = Instant::now();
        let mut c = ActivityClassifier::new();
        let none = InputEvidence::default();
        assert_eq!(c.observe(at, Some(0), none), ActivityState::Unknown);
        assert_eq!(c.observe(at + Duration::from_secs(7), Some(0), none), ActivityState::Unknown);
        assert_eq!(c.observe(at + Duration::from_secs(8), Some(0), none), ActivityState::Away);
    }

    #[test]
    fn one_missed_frame_does_not_flicker_working_to_unknown() {
        let at = Instant::now();
        let mut c = ActivityClassifier::new();
        assert_eq!(c.observe(at, Some(1), idle(0)), ActivityState::Unknown);
        assert_eq!(c.observe(at + Duration::from_millis(500), Some(1), idle(0)), ActivityState::Working);
        assert_eq!(c.observe(at + Duration::from_millis(550), Some(0), idle(0)), ActivityState::Working);
        assert_eq!(c.observe(at + Duration::from_millis(650), Some(1), idle(0)), ActivityState::Working);
        assert_eq!(c.observe(at + Duration::from_millis(1100), Some(1), idle(0)), ActivityState::Working);
    }

    #[test]
    fn persistent_face_loss_moves_to_unknown_then_away() {
        let at = Instant::now();
        let mut c = ActivityClassifier::new();
        c.observe(at, Some(1), idle(0));
        c.observe(at + Duration::from_millis(500), Some(1), idle(0));
        assert_eq!(c.observe(at + Duration::from_secs(1), Some(0), idle(1)), ActivityState::Working);
        assert_eq!(c.observe(at + Duration::from_secs(2), Some(0), idle(2)), ActivityState::Unknown);
        assert_eq!(c.observe(at + Duration::from_secs(9), Some(0), idle(9)), ActivityState::Away);
    }

    #[test]
    fn brief_false_face_does_not_cancel_away() {
        let at = Instant::now();
        let mut c = ActivityClassifier::new();
        let none = InputEvidence::default();
        c.observe(at, Some(0), none);
        assert_eq!(c.observe(at + Duration::from_secs(8), Some(0), none), ActivityState::Away);
        assert_eq!(c.observe(at + Duration::from_secs(9), Some(1), idle(0)), ActivityState::Away);
        assert_eq!(c.observe(at + Duration::from_secs(9) + Duration::from_millis(100), Some(0), idle(0)), ActivityState::Away);
        assert_eq!(c.observe(at + Duration::from_secs(10), Some(1), idle(0)), ActivityState::Away);
        assert_eq!(c.observe(at + Duration::from_secs(10) + Duration::from_millis(500), Some(1), idle(0)), ActivityState::Working);
    }

    #[test]
    fn returning_after_ten_seconds_idle_is_not_working() {
        let at = Instant::now();
        let mut c = ActivityClassifier::new();
        c.observe(at, Some(0), InputEvidence::default());
        c.observe(at + Duration::from_secs(8), Some(0), idle(10));
        assert_eq!(c.observe(at + Duration::from_secs(9), Some(1), idle(11)), ActivityState::Away);
        assert_eq!(c.observe(at + Duration::from_secs(9) + Duration::from_millis(500), Some(1), idle(11)), ActivityState::Unknown);
    }

    #[test]
    fn working_hysteresis_does_not_invent_new_input() {
        let start = Instant::now();
        let mut c = ActivityClassifier::new();
        c.observe(start, Some(1), idle(0));
        assert_eq!(
            c.observe(start + Duration::from_millis(500), Some(1), idle(0)),
            ActivityState::Working
        );
        // Every one-second "Idled" event is no longer an activity edge.
        assert_eq!(
            c.observe(start + Duration::from_secs(2), Some(1), idle(1)),
            ActivityState::Working
        );
        assert_eq!(
            c.observe(start + Duration::from_secs(3), Some(1), idle(4)),
            ActivityState::Working
        );
        assert_eq!(
            c.observe(start + Duration::from_secs(4), Some(1), idle(5)),
            ActivityState::Unknown
        );
        // Once Unknown, a four-second idle reading cannot re-enter Working.
        assert_eq!(
            c.observe(start + Duration::from_secs(5), Some(1), idle(4)),
            ActivityState::Unknown
        );
        assert_eq!(
            c.observe(start + Duration::from_secs(6), Some(1), idle(0)),
            ActivityState::Working
        );
        // Missing input evidence must not silently prolong Working.
        assert_eq!(
            c.observe(start + Duration::from_secs(7), Some(1), InputEvidence::default()),
            ActivityState::Unknown
        );
        assert_eq!(
            c.observe(start + Duration::from_secs(68), Some(1), idle(60)),
            ActivityState::Idling
        );
    }

    #[test]
    fn working_and_idling_require_explicit_os_evidence() {
        let at = Instant::now();
        let mut c = ActivityClassifier::new();
        c.observe(at, Some(1), InputEvidence::default());
        assert_eq!(c.observe(at + Duration::from_millis(500), Some(1), InputEvidence::default()), ActivityState::Unknown);
        assert_eq!(c.observe(at + Duration::from_millis(600), Some(1), idle(0)), ActivityState::Working);
        assert_eq!(c.observe(at + Duration::from_millis(700), Some(1), idle(1)), ActivityState::Working);
        assert_eq!(c.observe(at + Duration::from_millis(750), Some(1), idle(10)), ActivityState::Unknown);
        assert_eq!(c.observe(at + Duration::from_millis(800), Some(1), idle(60)), ActivityState::Idling);
        assert_eq!(c.observe(at + Duration::from_millis(900), Some(2), idle(0)), ActivityState::Unknown);
        assert_eq!(c.observe(at + Duration::from_millis(1000), None, idle(0)), ActivityState::Unknown);
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
