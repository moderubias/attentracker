use std::path::PathBuf;

use anyhow::Result;
use attent_camera::{AttentCamera, prepare_yunet};
use attent_model::{DetectorConfig, YuNetDetector, YuNetModel};
use nokhwa::utils::{CameraFormat, CameraIndex, FrameFormat, Resolution};

fn main() -> Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model_path = root.join("models/yunet.onnx");

    let model = YuNetModel::load(model_path)?;
    let detector = YuNetDetector::new(DetectorConfig::default())?;
    println!("YuNet loaded");

    let mut camera = AttentCamera::new(
        CameraIndex::Index(1),
        CameraFormat::new(Resolution::new(640, 480), FrameFormat::MJPEG, 30),
    )?;

    let frame = camera.capture()?;
    let input = prepare_yunet(&frame, 640)?;
    let output = model.infer(&input)?;
    let faces = detector.detect(&output)?;

    println!("Frame {}: {} face(s)", output.frame_sequence, faces.len());
    for (index, face) in faces.iter().enumerate() {
        println!(
            "Face {}: score={:.4}, bbox={:?}, landmarks={:?}",
            index + 1,
            face.confidence,
            face.bbox,
            face.landmarks,
        );
    }
    Ok(())
}
