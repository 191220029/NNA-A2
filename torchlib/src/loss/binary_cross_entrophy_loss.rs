use ndarray::ArrayD;

use crate::op::op::{AddScalar, EWiseAdd, Log, Mean, MulHadamard, Negate, Op};

use super::loss::Loss;

pub struct BinaryCrossEntrophyLoss {}

impl Loss for BinaryCrossEntrophyLoss {
    fn loss(
        &self,
        predicted: crate::tensor::tensor::TensorId,
        target: ndarray::ArrayD<f64>,
        factory: &mut crate::tensor::tensor_factory::TensorFactory,
    ) -> crate::tensor::tensor::TensorId {
        let target = factory.new_tensor(target, Some(false));
        let t = factory.make_from_op(
            Op::AddScalar(AddScalar { scalar: 1e-12 }),
            vec![predicted],
            None,
        );
        let t = factory.make_from_op(Op::Log(Log {}), vec![t], None);
        let t = factory.make_from_op(Op::MulHadamard(MulHadamard {}), vec![target, t], None);
        let y = factory.make_from_op(Op::Neg(Negate {}), vec![target], None);
        let ones = ArrayD::ones(factory.get(&y).unwrap().shape());
        let ones = factory.new_tensor(ones, Some(false));
        let y = factory.make_from_op(Op::EWiseAdd(EWiseAdd {}), vec![ones, y], None);
        let z = factory.make_from_op(Op::Neg(Negate {}), vec![predicted], None);
        let ones = ArrayD::ones(factory.get(&z).unwrap().shape());
        let ones = factory.new_tensor(ones, Some(false));
        let z = factory.make_from_op(Op::EWiseAdd(EWiseAdd {}), vec![ones, z], None);
        let z = factory.make_from_op(Op::AddScalar(AddScalar { scalar: 1e-12 }), vec![z], None);
        let z = factory.make_from_op(Op::Log(Log {}), vec![z], None);
        let y = factory.make_from_op(Op::MulHadamard(MulHadamard {}), vec![y, z], None);
        let t = factory.make_from_op(Op::EWiseAdd(EWiseAdd {}), vec![t, y], None);
        let loss = factory.make_from_op(Op::Mean(Mean {}), vec![t], None);
        loss
    }
}

impl BinaryCrossEntrophyLoss {
    pub fn new() -> Self {
        Self {}
    }
}

#[cfg(test)]
mod test_binary_cross_entrophy_loss {
    use ndarray::{ArrayD, IxDyn};

    use crate::{
        loss::{binary_cross_entrophy_loss::BinaryCrossEntrophyLoss, loss::Loss},
        tensor::tensor_factory::TensorFactory,
    };

    #[test]
    fn test_binary_cross_entrophy_loss() {
        let predicted =
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.0, 1.0, 0.1, 0.5, 1.0, 0.7]).unwrap();
        let target = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1., 0., 0., 0., 1., 0.]).unwrap();

        let mut factory = TensorFactory::default();
        let predicted = factory.new_tensor(predicted, None);

        let bcel = BinaryCrossEntrophyLoss::new();
        let loss = bcel.loss(predicted, target, &mut factory);

        assert_eq!(
            "[-13.7692071608162]",
            factory
                .get(&loss)
                .unwrap()
                .cached_data
                .as_ref()
                .unwrap()
                .to_string()
        );

        factory.backward(&loss, None, Some(true));
        assert_eq!(
            "[1]",
            factory
                .get(&loss)
                .unwrap()
                .grad
                .as_ref()
                .unwrap()
                .to_string()
        );
    }
}
