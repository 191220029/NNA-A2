use crate::tensor::tensor_factory::TensorFactory;

pub trait Optimizer {
    fn step(&mut self, factory: &mut TensorFactory);
    fn reset_grad(&mut self, factory: &mut TensorFactory);
}
