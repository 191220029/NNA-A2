use crate::{
    op::op::{EWiseAdd, Op},
    tensor::{tensor::TensorId, tensor_factory::TensorFactory},
};

use super::Module;

pub struct Residual {
    train: bool,
    child: Box<dyn Module>,
}

impl Module for Residual {
    fn init(&mut self) {
        self.train = true;
        self.child.init();
    }

    fn forward(&mut self, x: TensorId, factory: &mut TensorFactory) -> TensorId {
        let t = self.child.forward(x, factory);
        factory.make_from_op(Op::EWiseAdd(EWiseAdd {}), vec![x, t], None)
    }

    fn eval(&mut self) {
        self.train = false;
        self.child.eval();
    }

    fn train(&mut self) {
        self.train = true;
        self.child.train();
    }
}

impl Residual {
    pub fn new(child: Box<dyn Module>) -> Self {
        Self { train: true, child }
    }
}
