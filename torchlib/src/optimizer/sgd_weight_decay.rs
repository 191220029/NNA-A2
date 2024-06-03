use crate::tensor::{tensor::TensorId, tensor_factory::TensorFactory};

use super::optimizer::Optimizer;

pub struct SGDWeightDecay {
    params: Vec<TensorId>,
    lr: f64,
    weight_decay: f64,
}

impl Optimizer for SGDWeightDecay {
    fn step(&mut self, factory: &mut TensorFactory) {
        for param in &self.params {
            if self.weight_decay > 0. {
                factory.get_mut(param).unwrap().grad = Some(
                    factory.get(param).unwrap().grad.as_ref().unwrap()
                        + self.weight_decay
                            * factory.get(param).unwrap().cached_data.as_ref().unwrap(),
                );
            }
            let t = factory.get_mut(param).unwrap();
            let grad = t.grad.as_ref().unwrap();
            let data = t.cached_data.as_ref().unwrap();
            t.cached_data = Some(data - grad * self.lr);
        }
    }

    fn reset_grad(&mut self, factory: &mut TensorFactory) {
        self.params.iter().for_each(|p| {
            factory.get_mut(p).unwrap().grad = None;
        })
    }
}

impl SGDWeightDecay {
    pub fn new(params: Vec<TensorId>, lr: f64, weight_decay: Option<f64>) -> Self {
        Self {
            params,
            lr,
            weight_decay: match weight_decay {
                Some(w) => w,
                None => 0.,
            },
        }
    }
}
