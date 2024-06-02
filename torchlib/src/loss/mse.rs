use ndarray::ArrayD;

use crate::{
    op::op::{EWiseAdd, Negate, Op, PowerScalar, Summation},
    tensor::{tensor::TensorId, tensor_factory::TensorFactory},
};

use super::loss::Loss;

pub struct MSELoss {}

impl Loss for MSELoss {
    fn loss(
        &self,
        predicted: TensorId,
        target: ArrayD<f64>,
        factory: &mut TensorFactory,
    ) -> TensorId {
        let target = factory.new_tensor(target, None);
        let t = factory.make_from_op(Op::Neg(Negate {}), vec![target], None);
        let t = factory.make_from_op(Op::EWiseAdd(EWiseAdd {}), vec![predicted, t], None);
        let t = factory.make_from_op(Op::Power(PowerScalar {scalar: 2.}), vec![t], None);
        factory.make_from_op(Op::Sum(Summation { axis: None }), vec![t], None)
    }
}

impl MSELoss {
    pub fn new() -> Self {
        Self {}
    }
}
