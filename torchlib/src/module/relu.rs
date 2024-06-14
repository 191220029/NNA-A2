use crate::{
    op::op::MaxScalar,
    tensor::{tensor::TensorId, tensor_factory::TensorFactory},
};

use super::Module;

pub struct ReLU {
    train: bool,
}

impl Module for ReLU {
    fn init(&mut self) {
        self.train = true;
    }

    fn forward(&mut self, x: TensorId, factory: &mut TensorFactory) -> TensorId {
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
