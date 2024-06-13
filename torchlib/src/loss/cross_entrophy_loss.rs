use crate::{
    op::op::{DivScalar, DivTensor, Exp, MaxScalar, Negate, Op, Summation},
    tensor::{tensor::TensorId, tensor_factory::TensorFactory},
};

use super::loss::Loss;

pub struct CrossEntrophyLoss {}

impl Loss for CrossEntrophyLoss {
    fn loss(
        &self,
        predicted: TensorId,
        target: ndarray::ArrayD<f64>,
        factory: &mut TensorFactory,
    ) -> TensorId {
        let x = factory.make_from_op(Op::Exp(Exp {}), vec![predicted], None);
        let y = factory.make_from_op(Op::Exp(Exp {}), vec![predicted], None);
        let y = factory.make_from_op(Op::Sum(Summation { axis: None }), vec![y], None);
        let p = factory.make_from_op(Op::DivTensor(DivTensor::default()), vec![y], None);

        todo!()
    }
}
