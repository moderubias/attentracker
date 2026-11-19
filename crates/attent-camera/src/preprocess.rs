use std::time::{Instant, SystemTime};

use anyhow::{Context, Result, ensure};
use image::{RgbImage, imageops::{FilterType, resize}};
use crate::camera::CapturedFrame;

/// Geometry of a resized, centered image inside a square model input.
#[derive(Clone, Copy, Debug)]
pub struct Letterbox {
    pub source_width: u32,
    pub source_height: u32,
    pub input_side: u32,
    pub resized_width: u32,
    pub resized_height: u32,
    pub pad_left: u32,
    pub pad_top: u32,
}

impl Letterbox {
    /// Map an un-clipped point in model input coordinates back to the original image.
    /// Values may lie outside the source image when detections include padding.
    pub fn to_source(self, x: f32, y: f32) -> (f32, f32) {
        (
            (x - self.pad_left as f32) * self.source_width as f32 / self.resized_width as f32,
            (y - self.pad_top as f32) * self.source_height as f32 / self.resized_height as f32,
        )
    }
}

/// Exactly one YuNet input. BGR, NCHW, f32 in 0..=255, zero-padded.
pub struct YuNetInput {
    pub data: Vec<f32>,
    pub shape: [usize; 4],
    pub transform: Letterbox,
    pub frame_sequence: u64,
    pub received_at: SystemTime,
    pub received_tick: Instant,
}

pub fn prepare_yunet(frame: &CapturedFrame, input_side: u32) -> Result<YuNetInput> {
    let (data, transform) = preprocess_rgb(&frame.rgb, input_side)?;
    Ok(YuNetInput {
        data,
        shape: [1, 3, input_side as usize, input_side as usize],
        transform,
        frame_sequence: frame.meta.sequence,
        received_at: frame.meta.received_at,
        received_tick: frame.meta.received_tick,
    })
}

fn preprocess_rgb(rgb: &RgbImage, side: u32) -> Result<(Vec<f32>, Letterbox)> {
    let (w, h) = rgb.dimensions();
    ensure!(w > 0 && h > 0, "empty input image");
    ensure!(side > 0, "input side must be positive");

    let scale = (side as f64 / w as f64).min(side as f64 / h as f64);
    let new_w = ((w as f64 * scale).round() as u32).clamp(1, side);
    let new_h = ((h as f64 * scale).round() as u32).clamp(1, side);
    let pad_left = (side - new_w) / 2;
    let pad_top = (side - new_h) / 2;

    let s = side as usize;
    let plane_size = s.checked_mul(s).context("input dimensions overflow")?;
    let len = plane_size.checked_mul(3).context("tensor dimensions overflow")?;

    let resized = resize(rgb, new_w, new_h, FilterType::Triangle);
    let mut data = vec![0.0f32; len];
    for (i, px) in resized.as_raw().chunks_exact(3).enumerate() {
        let x = i % new_w as usize + pad_left as usize;
        let y = i / new_w as usize + pad_top as usize;
        let offset = y * s + x;
        data[offset] = px[2] as f32;                  // B
        data[plane_size + offset] = px[1] as f32;     // G
        data[2 * plane_size + offset] = px[0] as f32; // R
    }

    Ok((data, Letterbox {
        source_width: w,
        source_height: h,
        input_side: side,
        resized_width: new_w,
        resized_height: new_h,
        pad_left,
        pad_top,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn bgr_planes_are_correct() {
        let source = RgbImage::from_pixel(2, 2, Rgb([11, 22, 33]));
        let (data, geo) = preprocess_rgb(&source, 2).unwrap();
        assert_eq!(&data[0..4], &[33.0; 4]);
        assert_eq!(&data[4..8], &[22.0; 4]);
        assert_eq!(&data[8..12], &[11.0; 4]);
        assert_eq!(geo.to_source(1.0, 1.0), (1.0, 1.0));
    }

    #[test]
    fn letterbox_has_correct_padding_and_inverse_transform() {
        let source = RgbImage::from_pixel(4, 2, Rgb([10, 20, 30]));
        let (data, geo) = preprocess_rgb(&source, 4).unwrap();
        assert_eq!((geo.resized_width, geo.resized_height), (4, 2));
        assert_eq!((geo.pad_left, geo.pad_top), (0, 1));
        assert_eq!(data[0], 0.0);           // top padding
        assert_eq!(data[4], 30.0);          // B at x=0,y=1
        assert_eq!(geo.to_source(3.0, 2.0), (3.0, 1.0));
    }

    #[test]
    fn zero_side_is_rejected() {
        let source = RgbImage::from_pixel(4, 2, Rgb([10, 20, 30]));
        assert!(preprocess_rgb(&source, 0).is_err());
    }
}

