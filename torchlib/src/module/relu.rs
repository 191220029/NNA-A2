use crate::op::op::MaxScalar;

use super::Module;

pub struct ReLU {
    train: bool,
}

impl Module for ReLU {
    fn init(&mut self) {
        self.train = true;
    }

    fn parameters(&self) -> Vec<crate::tensor::tensor::TensorId> {
        vec![]
    }

    fn children(&self) -> Vec<Box<&dyn Module>> {
        vec![]
    }

    fn forward(
        &mut self,
        x: ndarray::ArrayD<f64>,
        factory: &mut crate::tensor::tensor_factory::TensorFactory,
    ) -> crate::tensor::tensor::TensorId {
        let x = factory.new_tensor(x, None);
        factory.make_from_op(
            crate::op::op::Op::Max(MaxScalar { scalar: 0. }),
            vec![x],
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

impl ReLU {
    pub fn new() -> Self {
        Self { train: true }
    }
}
