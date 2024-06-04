use crate::{
    op::op::{AddScalar, Exp, Negate, PowerScalar},
    tensor::tensor::TensorId,
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
        x: ndarray::ArrayD<f64>,
        factory: &mut crate::tensor::tensor_factory::TensorFactory,
    ) -> TensorId {
        let x = factory.new_tensor(x, None);
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
        let mut model = Sigmoid::new();
        let t = model.forward(x, factory);
        assert_eq!(
            "",
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
