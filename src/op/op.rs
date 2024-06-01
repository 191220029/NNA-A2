use ndarray::{ArrayD, Axis, IxDyn};

use crate::tensor::{tensor::Tensor, tensor_factory::TensorFactory};

#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    EWiseAdd(EWiseAdd),
    Sum(Summation),
}

impl Op {
    pub fn compute(&self, args: Vec<ArrayD<f64>>) -> ArrayD<f64> {
        match self {
            Op::EWiseAdd(e) => e.compute(args),
            Op::Sum(s) => s.compute(args),
        }
    }
    pub fn gradient(
        &self,
        out_grad: &ArrayD<f64>,
        node: &Tensor,
        factory: &TensorFactory,
    ) -> Vec<ArrayD<f64>> {
        match self {
            Op::EWiseAdd(e) => e.gradient(out_grad, node, factory),
            Op::Sum(s) => s.gradient(out_grad, node, factory),
        }
    }
}

trait OpTrait {
    fn compute(&self, args: Vec<ArrayD<f64>>) -> ArrayD<f64>;
    fn gradient(
        &self,
        out_grad: &ArrayD<f64>,
        node: &Tensor,
        factory: &TensorFactory,
    ) -> Vec<ArrayD<f64>>;
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct EWiseAdd {}
impl OpTrait for EWiseAdd {
    fn compute(&self, args: Vec<ArrayD<f64>>) -> ArrayD<f64> {
        &args[0] + &args[1]
    }
    fn gradient(&self, out_grad: &ArrayD<f64>, _: &Tensor, _: &TensorFactory) -> Vec<ArrayD<f64>> {
        vec![out_grad.clone(), out_grad.clone()]
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Summation {
    pub axis: Option<usize>,
}
impl OpTrait for Summation {
    fn compute(&self, args: Vec<ArrayD<f64>>) -> ArrayD<f64> {
        match self.axis {
            Some(axis) => args[0].sum_axis(Axis(axis)),
            None => ArrayD::from_elem(IxDyn(&[1]), args[0].sum()),
        }
    }
    fn gradient(
        &self,
        out_grad: &ArrayD<f64>,
        node: &Tensor,
        factory: &TensorFactory,
    ) -> Vec<ArrayD<f64>> {
        let input_shape = factory.get(&node.inputs[0]).unwrap().shape();
        let mut shape = out_grad.shape().to_vec();

        if let Some(axis) = self.axis {
            shape.insert(axis, 1);
        }

        let out_grad = out_grad.to_shared().reshape(shape).to_owned();
        vec![out_grad.broadcast(input_shape).unwrap().to_owned()]
    }
}
