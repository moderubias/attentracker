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
const RECENT_INPUT: Duration = Duration::from_secs(2);

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
                self.present_since = None;
                let absent_since = *self.absent_since.get_or_insert(now);
                let elapsed = now.saturating_duration_since(absent_since);
                if elapsed >= AWAY_AFTER {
                    ActivityState::Away
                } else if elapsed < FACE_LOSS_GRACE {
                    // Suppress individual missed detections without claiming a
                    // fresh input event or resetting a sustained Away state.
                    self.state
                } else {
                    ActivityState::Unknown
                }
            }
            Some(1) => {
                self.absent_since = None;
                let present_since = *self.present_since.get_or_insert(now);
                if now.saturating_duration_since(present_since) < FACE_CONFIRM {
                    // In particular, a one-frame false positive must not
                    // change Away into Working.
                    if self.state == ActivityState::Away {
                        ActivityState::Away
                    } else {
                        ActivityState::Unknown
                    }
                } else {
                    match input.idle_for {
                        Some(age) if age >= IDLING_AFTER => ActivityState::Idling,
                        Some(age) if age <= RECENT_INPUT => ActivityState::Working,
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
        assert_eq!(c.observe(at + Duration::from_millis(650), Some(1), idle(0)), ActivityState::Unknown);
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
    fn working_and_idling_require_explicit_os_evidence() {
        let at = Instant::now();
        let mut c = ActivityClassifier::new();
        c.observe(at, Some(1), InputEvidence::default());
        assert_eq!(c.observe(at + Duration::from_millis(500), Some(1), InputEvidence::default()), ActivityState::Unknown);
        assert_eq!(c.observe(at + Duration::from_millis(600), Some(1), idle(0)), ActivityState::Working);
        assert_eq!(c.observe(at + Duration::from_millis(700), Some(1), idle(10)), ActivityState::Unknown);
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
