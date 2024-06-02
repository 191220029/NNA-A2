use crate::tensor::{tensor::TensorId, tensor_factory::TensorFactory};

mod initialize;
pub mod linear;

trait Module {
    fn init(&mut self);

    fn parameters(&self) -> Vec<TensorId>;

    fn children(&self) -> Vec<Box<&dyn Module>>;

    fn forward(&mut self, x: TensorId, factory: &mut TensorFactory) -> TensorId;

    fn eval(&mut self);

    fn train(&mut self);
}
