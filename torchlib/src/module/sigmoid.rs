use crate::{
    op::op::{AddScalar, Exp, Negate, PowerScalar},
    tensor::{tensor::TensorId, tensor_factory::TensorFactory},
};

use super::Module;

pub struct Sigmoid {
    train: bool,
}

impl Module for Sigmoid {
    fn init(&mut self) {
        self.train = true;
    }

    fn forward(
        &mut self,
        x: TensorId,
        factory: &mut TensorFactory,
    ) -> TensorId {
        let t = factory.make_from_op(crate::op::op::Op::Neg(Negate {}), vec![x], None);
        let t = factory.make_from_op(crate::op::op::Op::Exp(Exp {}), vec![t], None);
        let t = factory.make_from_op(
            crate::op::op::Op::AddScalar(AddScalar { scalar: 1. }),
            vec![t],
            None,
        );
        factory.make_from_op(
            crate::op::op::Op::Power(PowerScalar { scalar: -1. }),
            vec![t],
            None,
        )
    }

    fn eval(&mut self) {
        self.train = false;
    }

    fn train(&mut self) {
        self.train = true;
    }
}

impl Sigmoid {
    pub fn new() -> Self {
        Self { train: true }
    }
}

#[cfg(test)]
mod test_soft_max {
    use ndarray::{ArrayD, IxDyn};

    use crate::{
        module::{sigmoid::Sigmoid, Module},
        tensor::tensor_factory::TensorFactory,
    };

    #[test]
    fn test_soft_max() {
        let factory = &mut TensorFactory::default();
        let data = vec![-5., -2., 0., 2., 5.];
        let x = ArrayD::from_shape_vec(IxDyn(&[1, 5]), data).unwrap();
        let x = factory.new_tensor(x, None);
        let mut model = Sigmoid::new();
        let t = model.forward(x, factory);
        assert_eq!(
            "[[0.0066928509242848554, 0.11920292202211755, 0.5, 0.8807970779778823, 0.9933071490757153]]",
            factory
                .get(&t)
                .unwrap()
                .cached_data
                .as_ref()
                .unwrap()
                .to_string()
        );
        factory.backward(&t, None, None);
    }
}
