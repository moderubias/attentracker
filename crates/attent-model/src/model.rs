use std::{
    path::Path,
    time::{Instant, SystemTime},
};

use anyhow::{Result, ensure};
use attent_camera::{Letterbox, YuNetInput};
use tract::prelude::*;

pub struct YuNetModel {
    runtime: Runnable,
}

#[derive(Debug)]
pub struct OutputTensor {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

#[derive(Debug)]
pub struct YuNetOutput {
    pub tensors: Vec<OutputTensor>,
    pub transform: Letterbox,
    pub frame_sequence: u64,
    pub received_at: SystemTime,
    pub received_tick: Instant,
}

impl YuNetModel {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let runtime = tract::onnx()?.load(path)?.into_model()?.into_runnable()?;

        Ok(Self { runtime })
    }

    pub fn infer(&self, input: &YuNetInput) -> Result<YuNetOutput> {
        ensure!(
            input.shape == [1, 3, 640, 640],
            "Unexpected YuNet input shape: {:?}",
            input.shape
        );

        ensure!(
            input.data.len() == input.shape.iter().product::<usize>(),
            "Input tensor length does not match shape"
        );

        // Borrow f32 as bytes; Tract constructs its own tensor.
        let bytes = bytemuck::cast_slice::<f32, u8>(&input.data);

        let tensor = Tensor::from_bytes(DatumType::F32, &input.shape, bytes)?;

        let outputs = self.runtime.run([tensor])?;

        ensure!(
            outputs.len() == 12,
            "Expected 12 YuNet outputs, got {}",
            outputs.len()
        );

        let mut tensors = Vec::with_capacity(12);

        for output in outputs {
            let (dtype, shape, bytes) = output.as_bytes()?;

            ensure!(
                dtype == DatumType::F32,
                "Unexpected YuNet output type: {dtype:?}"
            );

            ensure!(bytes.len() % 4 == 0, "Invalid f32 tensor byte length");

            let values = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            tensors.push(OutputTensor {
                shape: shape.to_vec(),
                values,
            });
        }

        Ok(YuNetOutput {
            tensors,
            transform: input.transform,
            frame_sequence: input.frame_sequence,
            received_at: input.received_at,
            received_tick: input.received_tick,
        })
    }
}

impl YuNetOutput {
    pub fn best_candidate_score(&self) -> Result<f32> {
        ensure!(self.tensors.len() == 12);

        let mut best = 0.0_f32;

        for i in 0..3 {
            let cls = &self.tensors[i].values;
            let obj = &self.tensors[i + 3].values;

            ensure!(cls.len() == obj.len(), "YuNet head length mismatch");

            for (&c, &o) in cls.iter().zip(obj) {
                let score = (c.clamp(0.0, 1.0) * o.clamp(0.0, 1.0)).sqrt();

                ensure!(score.is_finite(), "Non-finite YuNet score");

                best = best.max(score);
            }
        }

        Ok(best)
    }
}
