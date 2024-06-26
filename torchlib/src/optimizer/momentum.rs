use ndarray::ArrayD;

use crate::tensor::{tensor::TensorId, tensor_factory::TensorFactory};

use super::optimizer::Optimizer;

pub struct Momentum {
    params: Vec<TensorId>,
    lr: f64,
    momentum: f64,
    nesterov: bool,
    velocity: Vec<ArrayD<f64>>,
}

impl Optimizer for Momentum {
    fn step(&mut self, factory: &mut TensorFactory) {
        for (i, param) in self.params.iter().enumerate() {
            if self.nesterov {
                let prev_velocity = &self.velocity[i];
                let new_velocity = self.momentum * prev_velocity
                    - self.lr * factory.get(&param).unwrap().grad.as_ref().unwrap();
                factory.get_mut(&param).unwrap().cached_data = Some(
                    factory.get(&param).unwrap().cached_data.as_ref().unwrap()
                        + -self.momentum * prev_velocity
                        + (1. + self.momentum) * &new_velocity,
                );
                self.velocity[i] = new_velocity;
            } else {
                self.velocity[i] = self.momentum * &self.velocity[i]
                    - self.lr * factory.get(&param).unwrap().grad.as_ref().unwrap();
                factory.get_mut(&param).unwrap().cached_data = Some(
                    factory
                        .get_mut(&param)
                        .unwrap()
                        .cached_data
                        .as_ref()
                        .unwrap()
                        + &self.velocity[i],
                )
            }
        }
        factory.clean_all(self.params.clone());
        self.reset_grad(factory);
    }

    fn reset_grad(&mut self, factory: &mut TensorFactory) {
        self.params.iter().for_each(|p| {
            factory.get_mut(p).unwrap().grad = None;
        })
    }

    fn update_lr(&mut self, lr: Vec<f64>) {
        self.lr = lr[0];
    }

    fn lr(&self) -> Vec<f64> {
        vec![self.lr]
    }
}

impl Momentum {
    pub fn new(
        params: Vec<TensorId>,
        lr: f64,
        momentum: Option<f64>,
        nesterov: Option<bool>,
        factory: &TensorFactory,
    ) -> Self {
        let velocity = params
            .iter()
            .map(|p| ArrayD::zeros(factory.get(p).unwrap().shape()))
            .collect();
        Self {
            params,
            lr,
            momentum: momentum.unwrap_or(0.9),
            nesterov: nesterov.unwrap_or(false),
            velocity,
        }
    }
}
