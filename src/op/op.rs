use ndarray::{ArrayD, Axis, IxDyn};
use peroxide::fuga::{Matrix, Shape::Row};

use crate::tensor::{tensor::Tensor, tensor_factory::TensorFactory};

#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    EWiseAdd(EWiseAdd),
    Sum(Summation),
    MatMul(MatrixMul),
    BCast(BroadCast),
}

impl Op {
    pub fn compute(&self, args: Vec<ArrayD<f64>>) -> ArrayD<f64> {
        match self {
            Op::EWiseAdd(e) => e.compute(args),
            Op::Sum(s) => s.compute(args),
            Op::MatMul(m) => m.compute(args),
            Op::BCast(b) => b.compute(args),
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
            Op::MatMul(m) => m.gradient(out_grad, node, factory),
            Op::BCast(b) => b.gradient(out_grad, node, factory),
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
pub struct MatrixMul {}
impl OpTrait for MatrixMul {
    fn compute(&self, args: Vec<ArrayD<f64>>) -> ArrayD<f64> {
        from_matrix(into_matrix(args[0].to_owned()) * (into_matrix(args[1].to_owned())))
    }
    fn gradient(
        &self,
        out_grad: &ArrayD<f64>,
        node: &Tensor,
        factory: &TensorFactory,
    ) -> Vec<ArrayD<f64>> {
        let a = factory.get(&node.inputs[0]).unwrap();
        let b = factory.get(&node.inputs[1]).unwrap();
        let grad_a = from_matrix(
            into_matrix(out_grad.to_owned())
                * into_matrix(b.cached_data.clone().unwrap().t().to_owned()),
        );
        let grad_b = from_matrix(
            into_matrix(a.cached_data.clone().unwrap().t().to_owned())
                * into_matrix(out_grad.to_owned()),
        );
        return vec![grad_a, grad_b];
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

#[derive(Clone, Debug, PartialEq, Default)]
pub struct BroadCast {
    pub shape: Vec<usize>,
}
impl OpTrait for BroadCast {
    fn compute(&self, args: Vec<ArrayD<f64>>) -> ArrayD<f64> {
        args[0].broadcast(IxDyn(&self.shape)).unwrap().to_owned()
    }
    fn gradient(
        &self,
        out_grad: &ArrayD<f64>,
        node: &Tensor,
        factory: &TensorFactory,
    ) -> Vec<ArrayD<f64>> {
        let input_shape = factory.get(&node.inputs[0]).unwrap().shape();
        let mut grad = out_grad.clone();

        // 将广播后的梯度还原到原始形状
        for (axis, dim) in input_shape.iter().enumerate() {
            if *dim == 1 {
                grad = grad.sum_axis(Axis(axis))
            }
        }

        return vec![grad];
    }
}

pub(crate) fn into_matrix(array: ArrayD<f64>) -> Matrix {
    let shape = array.shape().to_owned();
    Matrix {
        data: array.into_raw_vec(),
        row: shape[0],
        col: shape[1],
        shape: Row,
    }
}

pub(crate) fn from_matrix(matrix: Matrix) -> ArrayD<f64> {
    ArrayD::from_shape_vec(IxDyn(&[matrix.row, matrix.col]), matrix.data).unwrap()
}
