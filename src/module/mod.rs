use crate::tensor::tensor::TensorId;

mod initialize;
pub mod linear;

trait Module {
    fn init(&mut self);

    fn parameters(&self) -> Vec<TensorId>;

    fn children(&self) -> Vec<Box<&dyn Module>>;

    fn forward(&mut self);

    fn eval(&mut self);

    fn train(&mut self);
}
