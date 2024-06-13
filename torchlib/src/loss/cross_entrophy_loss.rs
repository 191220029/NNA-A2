use crate::{
    op::op::{
        AddScalar, DivTensor, EWiseAdd, Exp, GetMax, Log, MulHadamard, Negate, Op, Summation,
    },
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
        let axis = factory.get(&predicted).unwrap().shape().len() - 1;
        let m = factory.make_from_op(Op::GetMax(GetMax { axis }), vec![predicted], None);

        eprintln!("{}", factory.get(&m).unwrap().cached_data.as_ref().unwrap().to_string());

        let m = factory.make_from_op(Op::Neg(Negate {}), vec![m], None);
        let m = factory.make_from_op(Op::EWiseAdd(EWiseAdd {}), vec![predicted, m], None);
        let exps = factory.make_from_op(Op::Exp(Exp {}), vec![m], None);
        let sum_exps = factory.make_from_op(Op::Sum(Summation { axis: None }), vec![exps], None);
        let soft_max = factory.make_from_op(
            Op::DivTensor(DivTensor::default()),
            vec![exps, sum_exps],
            None,
        );
        let soft_max = factory.make_from_op(
            Op::AddScalar(AddScalar { scalar: 1e-12 }),
            vec![soft_max],
            None,
        );
        let log_soft_max = factory.make_from_op(Op::Log(Log {}), vec![soft_max], None);

        let target = factory.new_tensor(target, None);
        let t = factory.make_from_op(
            Op::MulHadamard(MulHadamard {}),
            vec![target, log_soft_max],
            None,
        );
        // let t = factory.make_from_op(Op::Sum(Summation { axis: None }), vec![t], None);
        let loss = factory.make_from_op(Op::Neg(Negate {}), vec![t], None);

        loss
    }
}

impl CrossEntrophyLoss {
    pub fn new() -> Self {
        Self {}
    }
}

#[cfg(test)]
mod test_cel {
    use ndarray::{ArrayD, IxDyn};

    use crate::{loss::loss::Loss, tensor::tensor_factory::TensorFactory};

    use super::CrossEntrophyLoss;

    #[test]
    fn test_cross_entrophy_loss() {
        let predicted =
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0, 1.0, 0.1, 0.5, 2.0, 0.3]).unwrap();
        let target = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1., 0., 0., 0., 1., 0.]).unwrap();

        let mut factory = TensorFactory::default();
        let predicted = factory.new_tensor(predicted, None);

        let cel = CrossEntrophyLoss::new();
        let loss = cel.loss(predicted, target, &mut factory);

        eprintln!(
            "loss={}",
            factory
                .get(&loss)
                .unwrap()
                .cached_data
                .as_ref()
                .unwrap()
                .to_string()
        );

        factory.backward(&loss, None, Some(true));
        println!(
            "{}",
            factory
                .get(&predicted)
                .unwrap()
                .grad
                .as_ref()
                .unwrap()
                .to_string()
        )
    }
}
