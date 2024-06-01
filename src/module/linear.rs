use crate::tensor::{tensor::TensorId, tensor_factory::TensorFactory};

pub struct Linear {
    in_features: usize,
    out_features: usize,
    weight: TensorId,
}

impl Linear {
    pub fn new(facory: &mut TensorFactory) -> Self {
        Self
    }
}