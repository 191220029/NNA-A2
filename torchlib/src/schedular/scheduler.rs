use crate::tensor::tensor_factory::TensorFactory;

pub trait Scheduler {
    fn step(&mut self, epoch: usize, factory: &mut TensorFactory);
    fn reset_grad(&mut self, factory: &mut TensorFactory);
}
