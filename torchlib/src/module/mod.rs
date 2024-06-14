use crate::tensor::{tensor::TensorId, tensor_factory::TensorFactory};

pub mod flatten;
pub mod initialize;
pub mod linear;
pub mod relu;
pub mod residual;
pub mod sequential;
pub mod sigmoid;
pub mod softmax;

pub trait Module {
    fn init(&mut self);

    fn parameters(&self) -> Vec<crate::tensor::tensor::TensorId> {
        vec![]
    }

    fn children(&self) -> Vec<&Box<dyn Module>> {
        vec![]
    }

    fn forward(&mut self, x: TensorId, factory: &mut TensorFactory) -> TensorId;

    fn eval(&mut self);

    fn train(&mut self);
}
