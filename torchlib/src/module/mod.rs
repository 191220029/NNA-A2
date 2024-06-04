use ndarray::ArrayD;

use crate::tensor::{tensor::TensorId, tensor_factory::TensorFactory};

pub mod flatten;
pub mod initialize;
pub mod linear;
pub mod relu;

pub trait Module {
    fn init(&mut self);

    fn parameters(&self) -> Vec<TensorId>;

    fn children(&self) -> Vec<Box<&dyn Module>>;

    fn forward(&mut self, x: ArrayD<f64>, factory: &mut TensorFactory) -> TensorId;

    fn eval(&mut self);

    fn train(&mut self);
}
