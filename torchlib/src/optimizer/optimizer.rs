use crate::tensor::tensor_factory::TensorFactory;

pub trait Optimizer {
    fn step(&mut self, factory: &mut TensorFactory);
    fn reset_grad(&mut self, factory: &mut TensorFactory);
    fn lr(&self) -> Vec<f64>;
    fn update_lr(&mut self, lr: Vec<f64>);
}
