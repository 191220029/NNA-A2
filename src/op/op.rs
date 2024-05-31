use ndarray::ArrayD;

use crate::tensor::tensor::Tensor;

#[derive(Clone, Debug, PartialEq, Hash)]
pub enum Op {
    EWiseAdd(EWiseAdd),
}

impl Op {
    pub fn compute(&self, args: Vec<ArrayD<f64>>) -> ArrayD<f64> {
        match self {
            Op::EWiseAdd(e) => e.compute(args),
        }
    }
    pub fn gradient(&self, out_grad: &ArrayD<f64>, node: &Tensor) -> Vec<ArrayD<f64>> {
        match self {
            Op::EWiseAdd(e) => e.gradient(out_grad, node),
        }
    }
}

trait OpTrait {
    fn compute(&self, args: Vec<ArrayD<f64>>) -> ArrayD<f64>;
    fn gradient(&self, out_grad: &ArrayD<f64>, node: &Tensor) -> Vec<ArrayD<f64>>;
}

#[derive(Clone, Debug, PartialEq, Hash, Default)]
pub struct EWiseAdd {}
impl OpTrait for EWiseAdd {
    fn compute(&self, args: Vec<ArrayD<f64>>) -> ArrayD<f64> {
        &args[0] + &args[1]
    }
    fn gradient(&self, out_grad: &ArrayD<f64>, _: &Tensor) -> Vec<ArrayD<f64>> {
        vec![out_grad.clone(), out_grad.clone()]
    }
}
