//! Ephemeral, opt-in diagnostic frames. Never persist or log camera pixels.
use anyhow::Result;
use attent_camera::CapturedFrame;
use attent_model::FaceDetection;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use image::{ExtendedColorType, codecs::jpeg::JpegEncoder, imageops::{FilterType, resize}};
use serde::Serialize;

const MAX_PREVIEW_WIDTH: u32 = 640;
const JPEG_QUALITY: u8 = 72;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FaceOverlay {
    pub bbox: [f32; 4],
    pub landmarks: [[f32; 2]; 5],
    pub confidence: f32,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FramePreview {
    /// Sequence number of the same frame used to calculate the overlays.
    pub sequence: u64,
    /// Original image dimensions. Overlays are expressed in these coordinates.
    pub width: u32,
    pub height: u32,
    /// An in-memory JPEG data URL; returned only when preview is opted into.
    pub data_url: String,
    pub faces: Vec<FaceOverlay>,
}

pub fn encode_preview(frame: &CapturedFrame, faces: &[FaceDetection]) -> Result<FramePreview> {
    let (width, height) = frame.rgb.dimensions();
    let mut jpeg = Vec::new();
    {
        let mut encoder = JpegEncoder::new_with_quality(&mut jpeg, JPEG_QUALITY);
        if width > MAX_PREVIEW_WIDTH {
            let scaled_height = ((height as u64 * MAX_PREVIEW_WIDTH as u64 + width as u64 / 2)
                / width as u64).max(1) as u32;
            let scaled = resize(&frame.rgb, MAX_PREVIEW_WIDTH, scaled_height, FilterType::Triangle);
            encoder.encode(
                scaled.as_raw(),
                scaled.width(),
                scaled.height(),
                ExtendedColorType::Rgb8,
            )?;
        } else {
            encoder.encode(frame.rgb.as_raw(), width, height, ExtendedColorType::Rgb8)?;
        }
    }

    Ok(FramePreview {
        sequence: frame.meta.sequence,
        width,
        height,
        data_url: format!("data:image/jpeg;base64,{}", STANDARD.encode(&jpeg)),
        faces: faces
            .iter()
            .map(|f| FaceOverlay {
                bbox: f.bbox,
                landmarks: f.landmarks,
                confidence: f.confidence,
            })
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use attent_camera::CaptureMeta;
    use image::{Rgb, RgbImage};
    use nokhwa::utils::{CameraFormat, CameraIndex, FrameFormat, Resolution};
    use std::time::{Duration, Instant, SystemTime};

    #[test]
    fn preview_encodes_jpeg_and_keeps_face_geometry_in_original_pixels() {
        let frame = CapturedFrame {
            rgb: RgbImage::from_pixel(800, 600, Rgb([25, 80, 150])),
            meta: CaptureMeta {
                camera_index: CameraIndex::Index(1),
                stream_format: CameraFormat::new(Resolution::new(800, 600), FrameFormat::MJPEG, 30),
                sequence: 42,
                received_at: SystemTime::UNIX_EPOCH,
                received_tick: Instant::now(),
                driver_timestamp: None,
                source_format: FrameFormat::MJPEG,
                capture_duration: Duration::ZERO,
                decode_duration: Duration::ZERO,
            },
        };
        let detection = FaceDetection {
            bbox: [100.0, 80.0, 200.0, 250.0],
            landmarks: [[125.0, 130.0]; 5],
            confidence: 0.91,
        };
        let preview = encode_preview(&frame, &[detection]).unwrap();
        assert_eq!(preview.sequence, 42);
        assert_eq!((preview.width, preview.height), (800, 600));
        assert_eq!(preview.faces[0].bbox, detection.bbox);
        assert!(preview.data_url.starts_with("data:image/jpeg;base64,/9j/"));
    }
}
