pub mod detector;
pub mod model;

pub use detector::{DetectorConfig, FaceDetection, YuNetDetector};
pub use model::{OutputTensor, YuNetModel, YuNetOutput};
