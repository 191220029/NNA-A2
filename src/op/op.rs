use ndarray::ArrayD;

use crate::tensor::tensor::Tensor;

pub trait Op {
    fn compute(&self, args: Vec<ArrayD<f64>>) -> ArrayD<f64>;
    fn gradient(&self, out_grad: &Tensor, node: &Tensor) -> (Tensor, Tensor);
}

#[derive(Clone)]
pub struct EWiseAdd {}
impl Op for EWiseAdd {
    fn compute(&self, args: Vec<ArrayD<f64>>) -> ArrayD<f64> {
        &args[0] + &args[1]
    }
    fn gradient(&self, out_grad: &Tensor, node: &Tensor) -> (Tensor, Tensor) {
        (out_grad.clone(), out_grad.clone())
    }
}
