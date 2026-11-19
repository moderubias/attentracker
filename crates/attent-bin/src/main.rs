use std::path::PathBuf;

use anyhow::Result;

use attent_camera::{AttentCamera, prepare_yunet};

use attent_model::YuNetModel;

use nokhwa::utils::{CameraFormat, CameraIndex, FrameFormat, Resolution};

fn main() -> Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");

    let model_path = root.join("models/yunet.onnx");

    // Load model once.
    let model = YuNetModel::load(model_path)?;
    println!("YuNet loaded");

    // Open physical camera.
    let mut camera = AttentCamera::new(
        CameraIndex::Index(1),
        CameraFormat::new(Resolution::new(640, 480), FrameFormat::MJPEG, 30),
    )?;

    // Capture RGB frame.
    let frame = camera.capture()?;

    // Convert to BGR/NCHW and letterbox.
    let input = prepare_yunet(&frame, 640)?;

    // Run inference.
    let output = model.infer(&input)?;

    println!("Frame: {}", output.frame_sequence);
    println!(
        "Best candidate score: {:.4}",
        output.best_candidate_score()?
    );

    for (i, tensor) in output.tensors.iter().enumerate() {
        println!("Output {i}: {:?}", tensor.shape);
    }

    Ok(())
}
