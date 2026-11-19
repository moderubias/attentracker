use anyhow::{Result, ensure};

use crate::model::{OutputTensor, YuNetOutput};

const STRIDES: [usize; 3] = [8, 16, 32];

#[derive(Clone, Copy, Debug)]
pub struct DetectorConfig {
    pub score_threshold: f32,
    pub nms_threshold: f32,
    pub top_k: usize,
    pub max_faces: usize,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            score_threshold: 0.6,
            nms_threshold: 0.3,
            top_k: 5_000,
            max_faces: 32,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceDetection {
    /// Bounding rectangle in source-image pixels: [x, y, width, height].
    /// Clipped to the source image.
    pub bbox: [f32; 4],
    /// [right eye, left eye, nose, right mouth corner, left mouth corner].
    /// Landmarks are not clipped and may extend outside the source image.
    pub landmarks: [[f32; 2]; 5],
    pub confidence: f32,
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    bbox: [f32; 4], // model-space [x, y, width, height]
    landmarks: [[f32; 2]; 5],
    confidence: f32,
}

pub struct YuNetDetector {
    config: DetectorConfig,
}

impl YuNetDetector {
    pub fn new(config: DetectorConfig) -> Result<Self> {
        ensure!(
            config.score_threshold.is_finite() && (0.0..=1.0).contains(&config.score_threshold),
            "score_threshold must be finite and in [0, 1]"
        );
        ensure!(
            config.nms_threshold.is_finite() && (0.0..=1.0).contains(&config.nms_threshold),
            "nms_threshold must be finite and in [0, 1]"
        );
        ensure!(config.top_k > 0, "top_k must be positive");
        ensure!(config.max_faces > 0, "max_faces must be positive");
        Ok(Self { config })
    }

    pub fn detect(&self, output: &YuNetOutput) -> Result<Vec<FaceDetection>> {
        let geo = output.transform;
        let side = geo.input_side as usize;
        ensure!(matches!(side, 160 | 320 | 640), "unsupported YuNet input side: {side}");
        ensure!(geo.source_width > 0 && geo.source_height > 0);
        ensure!(geo.resized_width > 0 && geo.resized_height > 0);
        ensure!(output.tensors.len() == 12, "YuNet must return 12 tensors");

        let mut candidates = Vec::new();
        let mut invalid_head_scores = 0usize;
        let mut below_threshold = 0usize;
        let mut invalid_boxes = 0usize;
        let mut invalid_landmarks = 0usize;
        for (head, &stride) in STRIDES.iter().enumerate() {
            let cols = side / stride;
            let cells = cols * cols;
            let cls = &output.tensors[head];
            let obj = &output.tensors[head + 3];
            let bbox = &output.tensors[head + 6];
            let kps = &output.tensors[head + 9];

            validate_head(cls, cells, 1)?;
            validate_head(obj, cells, 1)?;
            validate_head(bbox, cells, 4)?;
            validate_head(kps, cells, 10)?;

            for idx in 0..cells {
                let class_score = cls.values[idx];
                let object_score = obj.values[idx];
                if !class_score.is_finite() || !object_score.is_finite() {
                    invalid_head_scores += 1;
                    continue;
                }
                let confidence =
                    (class_score.clamp(0.0, 1.0) * object_score.clamp(0.0, 1.0)).sqrt();
                if confidence < self.config.score_threshold {
                    below_threshold += 1;
                    continue;
                }

                let row = (idx / cols) as f32;
                let col = (idx % cols) as f32;
                let step = stride as f32;
                let b = &bbox.values[idx * 4..idx * 4 + 4];
                if !b.iter().all(|v| v.is_finite()) {
                    invalid_boxes += 1;
                    continue;
                }

                let cx = (col + b[0]) * step;
                let cy = (row + b[1]) * step;
                let w = b[2].exp() * step;
                let h = b[3].exp() * step;
                let rect = [cx - w / 2.0, cy - h / 2.0, w, h];
                if !rect.iter().all(|v| v.is_finite()) || w <= 0.0 || h <= 0.0 {
                    invalid_boxes += 1;
                    continue;
                }

                let points = &kps.values[idx * 10..idx * 10 + 10];
                if !points.iter().all(|v| v.is_finite()) {
                    invalid_landmarks += 1;
                    continue;
                }
                let mut landmarks = [[0.0; 2]; 5];
                for (i, xy) in landmarks.iter_mut().enumerate() {
                    xy[0] = (col + points[i * 2]) * step;
                    xy[1] = (row + points[i * 2 + 1]) * step;
                }
                if landmarks.iter().flatten().any(|v| !v.is_finite()) {
                    invalid_landmarks += 1;
                    continue;
                }

                candidates.push(Candidate {
                    bbox: rect,
                    landmarks,
                    confidence,
                });
            }
        }

        let candidates_before_top_k = candidates.len();
        candidates.sort_unstable_by(|a, b| b.confidence.total_cmp(&a.confidence));
        candidates.truncate(self.config.top_k);
        let mut nms_suppressed = 0usize;
        let mut clipped_out = 0usize;

        let mut selected: Vec<Candidate> = Vec::new();
        let mut faces = Vec::new();
        for candidate in candidates {
            if selected.iter().any(|other| {
                intersection_over_union(candidate.bbox, other.bbox) > self.config.nms_threshold
            }) {
                nms_suppressed += 1;
                continue;
            }
            selected.push(candidate);

            let (x1, y1) = geo.to_source(candidate.bbox[0], candidate.bbox[1]);
            let (x2, y2) = geo.to_source(
                candidate.bbox[0] + candidate.bbox[2],
                candidate.bbox[1] + candidate.bbox[3],
            );
            let max_x = geo.source_width as f32;
            let max_y = geo.source_height as f32;
            let x1 = x1.clamp(0.0, max_x);
            let y1 = y1.clamp(0.0, max_y);
            let x2 = x2.clamp(0.0, max_x);
            let y2 = y2.clamp(0.0, max_y);
            if x2 <= x1 || y2 <= y1 {
                clipped_out += 1;
                continue;
            }

            let mut landmarks = candidate.landmarks;
            for xy in &mut landmarks {
                (xy[0], xy[1]) = geo.to_source(xy[0], xy[1]);
            }
            faces.push(FaceDetection {
                bbox: [x1, y1, x2 - x1, y2 - y1],
                landmarks,
                confidence: candidate.confidence,
            });
            if faces.len() >= self.config.max_faces {
                break;
            }
        }
        tracing::trace!(
            target: "attent_model::detector",
            frame_sequence = output.frame_sequence,
            candidate_cells = STRIDES.iter().map(|stride| (side / stride).pow(2)).sum::<usize>(),
            below_threshold,
            invalid_head_scores,
            invalid_boxes,
            invalid_landmarks,
            candidates_before_top_k,
            candidates_after_top_k = candidates_before_top_k.min(self.config.top_k),
            nms_suppressed,
            clipped_out,
            faces = faces.len(),
            score_threshold = self.config.score_threshold,
            nms_threshold = self.config.nms_threshold,
            "YuNet candidate filtering and NMS summary"
        );
        Ok(faces)
    }
}

fn validate_head(tensor: &OutputTensor, count: usize, channels: usize) -> Result<()> {
    ensure!(
        tensor.shape.as_slice() == &[1, count, channels],
        "unexpected YuNet head shape: {:?}, expected [1, {count}, {channels}]",
        tensor.shape
    );
    ensure!(
        tensor.values.len() == count * channels,
        "YuNet head buffer length does not match shape"
    );
    Ok(())
}

fn intersection_over_union(a: [f32; 4], b: [f32; 4]) -> f32 {
    let left = a[0].max(b[0]);
    let top = a[1].max(b[1]);
    let right = (a[0] + a[2]).min(b[0] + b[2]);
    let bottom = (a[1] + a[3]).min(b[1] + b[3]);
    let intersection = (right - left).max(0.0) * (bottom - top).max(0.0);
    let union = a[2] * a[3] + b[2] * b[3] - intersection;
    if union > 0.0 && union.is_finite() {
        intersection / union
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Instant, SystemTime};

    use attent_camera::Letterbox;

    use super::*;

    fn mock_output() -> YuNetOutput {
        mock_output_for_side(640)
    }

    fn mock_output_for_side(side: usize) -> YuNetOutput {
        let mut tensors = Vec::new();
        for group in 0..4 {
            for &stride in &STRIDES {
                let count = (side / stride).pow(2);
                let channels = [1, 1, 4, 10][group];
                tensors.push(OutputTensor {
                    shape: vec![1, count, channels],
                    values: vec![0.0; count * channels],
                });
            }
        }
        YuNetOutput {
            tensors,
            transform: Letterbox {
                source_width: 640,
                source_height: 480,
                input_side: side as u32,
                resized_width: side as u32,
                resized_height: side as u32 * 3 / 4,
                pad_left: 0,
                pad_top: (side as u32 - side as u32 * 3 / 4) / 2,
            },
            frame_sequence: 1,
            received_at: SystemTime::UNIX_EPOCH,
            received_tick: Instant::now(),
        }
    }

    #[test]
    fn decoding_restores_source_coordinates_and_landmarks() {
        let mut output = mock_output();
        let idx = 20 * 80 + 5; // stride=8, row=20, col=5
        output.tensors[0].values[idx] = 1.0;
        output.tensors[3].values[idx] = 0.81;
        output.tensors[6].values[idx * 4..idx * 4 + 4].copy_from_slice(&[
            0.5,
            0.5,
            4.0_f32.ln(),
            2.0_f32.ln(),
        ]);
        for n in 0..5 {
            output.tensors[9].values[idx * 10 + n * 2] = 0.5;
            output.tensors[9].values[idx * 10 + n * 2 + 1] = 0.5;
        }

        let detector = YuNetDetector::new(DetectorConfig::default()).unwrap();
        let faces = detector.detect(&output).unwrap();
        assert_eq!(faces.len(), 1);
        let face = faces[0];
        assert!((face.confidence - 0.9).abs() < 1e-5);
        for (actual, expected) in face.bbox.into_iter().zip([28.0, 76.0, 32.0, 16.0]) {
            assert!((actual - expected).abs() < 1e-4);
        }
        assert_eq!(face.landmarks, [[44.0, 84.0]; 5]);
    }

    #[test]
    fn nms_removes_overlapping_lower_scored_candidates() {
        let mut output = mock_output();
        for (col, confidence) in [(5, 1.0), (6, 0.8)] {
            let idx = 20 * 80 + col;
            output.tensors[0].values[idx] = confidence;
            output.tensors[3].values[idx] = 1.0;
            output.tensors[6].values[idx * 4..idx * 4 + 4].copy_from_slice(&[
                0.5,
                0.5,
                4.0_f32.ln(),
                2.0_f32.ln(),
            ]);
        }
        let detector = YuNetDetector::new(DetectorConfig::default()).unwrap();
        let faces = detector.detect(&output).unwrap();
        assert_eq!(faces.len(), 1);
        assert_eq!(faces[0].confidence, 1.0);

        let a = [0.0, 0.0, 10.0, 10.0];
        let b = [1.0, 1.0, 10.0, 10.0];
        let far = [30.0, 30.0, 10.0, 10.0];
        assert!(intersection_over_union(a, b) > 0.3);
        assert_eq!(intersection_over_union(a, far), 0.0);
    }

    #[test]
    fn smaller_model_outputs_map_back_to_source_space() {
        let mut output = mock_output_for_side(320);
        let idx = 10 * 40 + 5; // 320/8=40 cells per row
        output.tensors[0].values[idx] = 1.0;
        output.tensors[3].values[idx] = 1.0;
        output.tensors[6].values[idx * 4..idx * 4 + 4].copy_from_slice(&[
            0.5, 0.5, 2.0_f32.ln(), 2.0_f32.ln(),
        ]);
        let faces = YuNetDetector::new(DetectorConfig::default())
            .unwrap().detect(&output).unwrap();
        assert_eq!(faces.len(), 1);
        assert_eq!(faces[0].bbox, [72.0, 88.0, 32.0, 32.0]);
    }

    #[test]
    fn malformed_head_is_rejected() {
        let mut output = mock_output();
        output.tensors[8].shape = vec![1, 401, 4];
        let detector = YuNetDetector::new(DetectorConfig::default()).unwrap();
        assert!(detector.detect(&output).is_err());
    }

    #[test]
    fn invalid_threshold_is_rejected() {
        let config = DetectorConfig {
            nms_threshold: f32::NAN,
            ..DetectorConfig::default()
        };
        assert!(YuNetDetector::new(config).is_err());
    }
}
