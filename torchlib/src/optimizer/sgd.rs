use crate::tensor::{tensor::TensorId, tensor_factory::TensorFactory};

use super::optimizer::Optimizer;

pub struct SGD {
    params: Vec<TensorId>,
    lr: f64,
}

impl Optimizer for SGD {
    fn step(&self, factory: &mut TensorFactory) {
        self.params.iter().for_each(|p| {
            let t = factory.get_mut(p).unwrap();
            let grad = t.grad.clone().unwrap();
            let data = t.cached_data.clone().unwrap();
            t.cached_data = Some(data - grad * self.lr);
        });
    }

    fn reset_grad(&mut self, factory: &mut TensorFactory) {
        self.params.iter().for_each(|p| {
            factory.get_mut(p).unwrap().grad = None;
        })
    }
}

impl SGD {
    pub fn new(params: Vec<TensorId>, lr: f64) -> Self {
        Self { params, lr }
    }
}
