use std::{
    path::Path,
    time::{Instant, SystemTime},
};

use anyhow::{Result, ensure};
use attent_camera::{Letterbox, YuNetInput};
use tract::prelude::*;

pub struct YuNetModel {
    runtime: Runnable,
    input_side: u32,
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
        Self::load_for_side(path, 640)
    }

    /// Specialize the ONNX graph before Tract compiles its execution plan.
    /// YuNet output head sizes depend on the input side, so preprocessing and
    /// postprocessing must use exactly the same value.
    pub fn load_for_side(path: impl AsRef<Path>, input_side: u32) -> Result<Self> {
        ensure!(
            matches!(input_side, 160 | 320 | 640),
            "unsupported YuNet input side: {input_side}"
        );
        let begun = Instant::now();
        let mut model = tract::onnx()?.load(path)?;
        if input_side != 640 {
            let side = input_side as usize;
            let fact = format!("1,3,{side},{side},f32");
            model.set_input_fact(0, fact.as_str())?;
        }
        let runtime = model.into_model()?.into_runnable()?;
        tracing::info!(
            target: "attent_model::runtime",
            model = "YuNet",
            input_side,
            load_ms = begun.elapsed().as_secs_f64() * 1000.0,
            "ONNX model loaded"
        );
        Ok(Self { runtime, input_side })
    }

    pub fn infer(&self, input: &YuNetInput) -> Result<YuNetOutput> {
        ensure!(
            input.shape == [1, 3, self.input_side as usize, self.input_side as usize],
            "Unexpected YuNet input shape: {:?}",
            input.shape
        );

        ensure!(
            input.data.len() == input.shape.iter().product::<usize>(),
            "Input tensor length does not match shape"
        );

        // Borrow f32 as bytes; Tract constructs its own tensor.
        let tensor_start = Instant::now();
        let bytes = bytemuck::cast_slice::<f32, u8>(&input.data);
        let tensor = Tensor::from_bytes(DatumType::F32, &input.shape, bytes)?;
        let tensor_ms = tensor_start.elapsed().as_secs_f64() * 1000.0;

        let execution_start = Instant::now();
        let outputs = self.runtime.run([tensor])?;
        let execute_ms = execution_start.elapsed().as_secs_f64() * 1000.0;
        let unpack_start = Instant::now();

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

        tracing::trace!(
            target: "attent_model::inference",
            frame_sequence = input.frame_sequence,
            tensor_create_ms = tensor_ms,
            tract_execute_ms = execute_ms,
            outputs_copy_ms = unpack_start.elapsed().as_secs_f64() * 1000.0,
            "ONNX inference stages"
        );

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smaller_yunet_input_is_executable() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/yunet.onnx");
        let side = 320_u32;
        let model = YuNetModel::load_for_side(path, side)
            .expect("Tract must compile the model for a 320x320 input");
        let input = YuNetInput {
            data: vec![0.0_f32; 3 * side as usize * side as usize],
            shape: [1, 3, side as usize, side as usize],
            transform: Letterbox {
                source_width: 640,
                source_height: 480,
                input_side: side,
                resized_width: side,
                resized_height: 240,
                pad_left: 0,
                pad_top: 40,
            },
            frame_sequence: 1,
            received_at: SystemTime::UNIX_EPOCH,
            received_tick: Instant::now(),
        };
        let output = model.infer(&input).expect("320x320 inference must succeed");
        assert_eq!(output.tensors.len(), 12);
        assert_eq!(output.tensors[0].shape, vec![1, 40 * 40, 1]);
        assert_eq!(output.tensors[6].shape, vec![1, 40 * 40, 4]);
    }
}
