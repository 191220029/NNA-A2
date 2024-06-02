use ndarray::ArrayD;

use crate::tensor::{tensor::TensorId, tensor_factory::TensorFactory};

pub trait Loss {
    fn loss(
        &self,
        predicted: TensorId,
        target: ArrayD<f64>,
        factory: &mut TensorFactory,
    ) -> TensorId;
}
