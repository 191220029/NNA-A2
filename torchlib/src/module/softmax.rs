use crate::{
    op::op::{DivScalar, Exp, Summation},
    tensor::{tensor::TensorId, tensor_factory::TensorFactory},
};

use super::Module;

pub struct SoftMax {
    train: bool,
}

impl Module for SoftMax {
    fn init(&mut self) {
        self.train = true;
    }

    fn forward(&mut self, x: TensorId, factory: &mut TensorFactory) -> TensorId {
        let t = factory.make_from_op(crate::op::op::Op::Exp(Exp {}), vec![x], None);
        let s = factory.make_from_op(
            crate::op::op::Op::Sum(Summation { axis: None }),
            vec![t],
            None,
        );
        let t = factory.make_from_op(crate::op::op::Op::Exp(Exp {}), vec![x], None);
        let scalar = factory.get(&s).unwrap().cached_data.as_ref().unwrap().sum();
        factory.make_from_op(
            crate::op::op::Op::DivScalar(DivScalar { scalar }),
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

impl SoftMax {
    pub fn new() -> Self {
        Self { train: true }
    }
}

#[cfg(test)]
mod test_soft_max {
    use ndarray::{ArrayD, IxDyn};

    use crate::{module::Module, tensor::tensor_factory::TensorFactory};

    use super::SoftMax;

    #[test]
    fn test_soft_max() {
        let factory = &mut TensorFactory::default();
        let data = vec![-5., -2., 0., 2., 5.];
        let x = ArrayD::from_shape_vec(IxDyn(&[1, 5]), data).unwrap();
        let mut model = SoftMax::new();
        let x = factory.new_tensor(x, None);
        let t = model.forward(x, factory);
        assert_eq!("[[0.00004293209435280523, 0.0008623141663130462, 0.006371687749789713, 0.04708075822806539, 0.945642307761479]]", factory.get(&t).unwrap().cached_data.as_ref().unwrap().to_string());
        factory.backward(&t, None, None);
    }
}
