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
        // The upstream dynamic-shape YuNet needs concrete dimensions at
        // Tract plan compilation time, including the 640x640 baseline.
        // The old, fixed-640 graph cannot be changed merely by setting
        // its input fact: intermediate Conv shapes are hard-coded.
        let fact = format!("1,3,{input_side},{input_side},f32");
        model.set_input_fact(0, fact.as_str())?;
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
    fn dynamic_yunet_supports_all_selected_sizes() {
        // CI explicitly supplies this path after fetching and checksum-
        // verifying the upstream model. Offline `cargo test` remains usable.
        let Some(path) = std::env::var_os("ATTENT_TEST_DYNAMIC_YUNET") else {
            eprintln!("dynamic YuNet integration test skipped (set ATTENT_TEST_DYNAMIC_YUNET)");
            return;
        };
        let path = Path::new(&path);
        for side in [160_u32, 320, 640] {
            let model = YuNetModel::load_for_side(path, side)
                .unwrap_or_else(|err| panic!("load dynamic YuNet at {side}px failed: {err:#}"));
            let input = YuNetInput {
                data: vec![0.0_f32; 3 * side as usize * side as usize],
                shape: [1, 3, side as usize, side as usize],
                transform: Letterbox {
                    source_width: 640,
                    source_height: 480,
                    input_side: side,
                    resized_width: side,
                    resized_height: side * 3 / 4,
                    pad_left: 0,
                    pad_top: side / 8,
                },
                frame_sequence: 1,
                received_at: SystemTime::UNIX_EPOCH,
                received_tick: Instant::now(),
            };
            let output = model.infer(&input)
                .unwrap_or_else(|err| panic!("infer dynamic YuNet at {side}px failed: {err:#}"));
            assert_eq!(output.tensors.len(), 12);
            assert_eq!(output.tensors[0].shape, vec![1, (side as usize / 8).pow(2), 1]);
            assert_eq!(output.tensors[6].shape, vec![1, (side as usize / 8).pow(2), 4]);
        }
    }

    #[test]
    fn fixed_640_yunet_still_works_at_original_resolution() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/yunet.onnx");
        let model = YuNetModel::load(path).expect("the original 640px model must remain supported");
        assert_eq!(model.input_side, 640);
    }
}
