use anyhow::{Result, ensure};
use image::imageops::{FilterType, resize};
use nokhwa::{
    native_api_backend,
    query,
    Camera,
    pixel_format::RgbFormat,
    utils::{
        CameraIndex,
        RequestedFormat,
        RequestedFormatType,
    },
};
use tract::prelude::*;

use kernix::{Order, tensor::Tensor as KernixTensor};
use tract::prelude::{DatumType, Tensor as TractTensor};

const SIZE: u32 = 640;
const MODEL: &str = "models/yunet.onnx";

fn main() -> Result<()> {
    // 1. Load and prepare ONNX model
    let model = tract::onnx()?
        .load(MODEL)?
        .into_model()?
        .into_runnable()?;

    println!("YuNet loaded");

    // 2. Open camera
    let backend = native_api_backend().unwrap();

    let cameras = query(backend)?;

    println!("Available cameras:");

    for camera in &cameras {
        println!(
            "Index: {:?}, Name: {}",
            camera.index(),
            camera.human_name()
        );
    }

    // Select the actual capture device after inspecting the list.
    let camera_index = CameraIndex::Index(1);

    let request = RequestedFormat::new::<RgbFormat>(
        RequestedFormatType::AbsoluteHighestFrameRate,
    );

    let mut camera = Camera::new(camera_index, request)?;

    println!("Selected format: {:?}", camera.camera_format());

    camera.open_stream()?;

    // 3. Capture and decode RGB frame
    let frame = camera.frame()?;
    let rgb = frame.decode_image::<RgbFormat>()?;

    // Initial simple resize, without aspect-ratio preservation
    let resized = resize(
        &rgb,
        SIZE,
        SIZE,
        FilterType::Triangle,
    );

    // RGB -> BGR, HWC -> NCHW
    let side = SIZE as usize;
    let mut data = Vec::with_capacity(3 * side * side);

    for c in 0..3 {
        for y in 0..side {
            for x in 0..side {
                let pixel = resized.get_pixel(x as u32, y as u32);
                data.push(pixel.0[2 - c] as f32);
            }
        }
    }

    // 5. Execute model
    // Kernix owns and validates the tensor
    let input = KernixTensor::from_values(
        [1, 3, side, side],
        data,
    )?;

    // Boundary adapter: Kernix -> Tract
    let values = input.as_contiguous_slice(Order::C)?;
    let bytes = bytemuck::cast_slice(values);

    let tract_input = TractTensor::from_bytes(
        DatumType::F32,
        &[1, 3, side, side],
        bytes,
    )?;

    let outputs = model.run([tract_input])?;

    ensure!(
        outputs.len() == 12,
        "Expected 12 YuNet outputs, got {}",
        outputs.len()
    );

    // Official output order:
    // cls_8, cls_16, cls_32,
    // obj_8, obj_16, obj_32,
    // bbox_8, bbox_16, bbox_32,
    // kps_8, kps_16, kps_32

    let mut best_score = 0.0_f32;

    for stride_idx in 0..3 {
        let cls = outputs[stride_idx].as_slice::<f32>()?;
        let obj = outputs[stride_idx + 3].as_slice::<f32>()?;

        ensure!(cls.len() == obj.len(), "Head size mismatch");

        for (&c, &o) in cls.iter().zip(obj.iter()) {
            let score =
                (c.clamp(0.0, 1.0) * o.clamp(0.0, 1.0)).sqrt();

            best_score = best_score.max(score);
        }
    }

    println!("Best face candidate score: {best_score:.4}");
    println!("Face candidate above 0.9: {}", best_score >= 0.9);

    camera.stop_stream()?;

    Ok(())
}
