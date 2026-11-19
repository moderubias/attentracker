use kernix::Tensor as KernixTensor;
use tract::prelude::Tensor as TractTensor;
use 

pub struct AttentModel {
    input: Vec<f32>,
    output: Vec<TractTensor<f32>>,
    tensor: Tensor<f32>,
}

impl AttentModel {
    pub fn new(input: ) -> Self {
        let tensor = KernixTensor::<f32>::from_values(
            input.shape,
            input.data,
        )
    }
}
