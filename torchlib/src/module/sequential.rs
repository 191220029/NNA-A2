use crate::tensor::{tensor::TensorId, tensor_factory::TensorFactory};

use super::Module;

pub struct Sequential {
    train: bool,
    childs: Vec<Box<dyn Module>>,
}

impl Module for Sequential {
    fn init(&mut self) {
        self.train = true;
        self.childs.iter_mut().for_each(|c| {
            c.train();
        });
    }

    fn forward(&mut self, x: TensorId, factory: &mut TensorFactory) -> TensorId {
        let mut t = x;
        self.childs.iter_mut().for_each(|m| {
            t = m.forward(t, factory);
        });
        t
    }

    fn eval(&mut self) {
        self.train = false;
        self.childs.iter_mut().for_each(|c| {
            c.eval();
        });
    }

    fn train(&mut self) {
        self.train = true;
        self.childs.iter_mut().for_each(|c| {
            c.train();
        });
    }

    fn children(&self) -> Vec<&Box<dyn Module>> {
        self.childs.iter().map(|x| x).collect()
    }
}

impl Sequential {
    pub fn new(childs: Vec<Box<dyn Module>>) -> Self {
        let mut s = Self {
            train: true,
            childs: childs,
        };
        s.init();
        s
    }
}
