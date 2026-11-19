//! Conservative interpretation of observable evidence.
//!
//! Camera presence never proves productivity. `Working` only means one face
//! and *recent verified input*; it does not establish task relevance.
//! Without an OS input-idle provider, the state remains Unknown or (after a
//! sustained period without a visible face) Away.

use std::time::{Duration, Instant};
use serde::Serialize;

const AWAY_AFTER: Duration = Duration::from_secs(8);
const IDLING_AFTER: Duration = Duration::from_secs(60);
const RECENT_INPUT: Duration = Duration::from_secs(15);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivityState {
    #[default]
    Unknown,
    Working,
    Idling,
    Away,
}

/// Explicitly sourced OS-wide input idle estimate, if available.
/// None is not equivalent to no input.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InputEvidence {
    pub idle_for: Option<Duration>,
}

#[derive(Default)]
pub struct ActivityClassifier {
    absent_since: Option<Instant>,
}

impl ActivityClassifier {
    pub fn new() -> Self { Self::default() }

    /// `face_count=None` denotes no valid camera sample; don't interpret it
    /// as an absence. A person visible next to the camera is not necessarily
    /// the account owner.
    pub fn observe(
        &mut self, now: Instant, face_count: Option<usize>, input: InputEvidence,
    ) -> ActivityState {
        match face_count {
            None => {
                self.absent_since = None;
                ActivityState::Unknown
            }
            Some(0) => {
                let beginning = self.absent_since.get_or_insert(now);
                if now.saturating_duration_since(*beginning) >= AWAY_AFTER {
                    ActivityState::Away
                } else {
                    ActivityState::Unknown
                }
            }
            Some(1) => {
                self.absent_since = None;
                match input.idle_for {
                    Some(age) if age >= IDLING_AFTER => ActivityState::Idling,
                    Some(age) if age <= RECENT_INPUT => ActivityState::Working,
                    _ => ActivityState::Unknown,
                }
            }
            Some(_) => {
                self.absent_since = None;
                ActivityState::Unknown
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn away_requires_sustained_absence_and_resets_on_face() {
        let at = Instant::now();
        let mut c = ActivityClassifier::new();
        let none = InputEvidence::default();
        assert_eq!(c.observe(at, Some(0), none), ActivityState::Unknown);
        assert_eq!(c.observe(at + Duration::from_secs(7), Some(0), none), ActivityState::Unknown);
        assert_eq!(c.observe(at + Duration::from_secs(8), Some(0), none), ActivityState::Away);
        assert_eq!(c.observe(at + Duration::from_secs(9), Some(1), none), ActivityState::Unknown);
        assert_eq!(c.observe(at + Duration::from_secs(10), Some(0), none), ActivityState::Unknown);
    }

    #[test]
    fn working_idling_require_explicit_input_evidence() {
        let at = Instant::now();
        let mut c = ActivityClassifier::new();
        let input = |seconds| InputEvidence { idle_for: Some(Duration::from_secs(seconds)) };
        assert_eq!(c.observe(at, Some(1), InputEvidence::default()), ActivityState::Unknown);
        assert_eq!(c.observe(at, Some(1), input(2)), ActivityState::Working);
        assert_eq!(c.observe(at, Some(1), input(30)), ActivityState::Unknown);
        assert_eq!(c.observe(at, Some(1), input(61)), ActivityState::Idling);
        assert_eq!(c.observe(at, Some(2), input(2)), ActivityState::Unknown);
        assert_eq!(c.observe(at, None, input(61)), ActivityState::Unknown);
    }

    #[test]
    fn no_camera_face_is_not_proof_of_user_identity() {
        let now = Instant::now();
        let mut c = ActivityClassifier::new();
        let input = InputEvidence { idle_for: Some(Duration::ZERO) };
        assert_eq!(c.observe(now, Some(0), input), ActivityState::Unknown);
        assert_eq!(c.observe(now + Duration::from_secs(9), Some(0), input), ActivityState::Away);
    }
}
