use ndarray::ArrayD;

use crate::tensor::{tensor::TensorId, tensor_factory::TensorFactory};

use super::optimizer::Optimizer;

pub struct StepDecay {
    params: Vec<TensorId>,
    lr: f64,
    momentum: f64,
    nesterov: bool,
    velocity: Vec<ArrayD<f64>>,
}

impl Optimizer for StepDecay {
    fn step(&mut self, factory: &mut TensorFactory) {
        unimplemented!()
    }
    fn reset_grad(&mut self, factory: &mut TensorFactory) {
        unimplemented!();
        self.params.iter().for_each(|p| {
            factory.get_mut(p).unwrap().grad = None;
        })
    }
}

impl StepDecay {
    pub fn new(
        params: Vec<TensorId>,
        lr: f64,
        momentum: Option<f64>,
        nesterov: Option<bool>,
        factory: &TensorFactory,
    ) -> Self {
        unimplemented!();
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
