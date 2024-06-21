use crate::{optimizer::optimizer::Optimizer, tensor::tensor_factory::TensorFactory};

use super::scheduler::Scheduler;

pub struct StepDecay {
    optimizer: Box<dyn Optimizer>,
    step_size: usize,
    gamma: f64,
}

impl Scheduler for StepDecay {
    fn step(&mut self, epoch: usize, factory: &mut TensorFactory) {
        self.optimizer.step(factory);
        if epoch % self.step_size == 0 && epoch != 0 {
            self.optimizer
                .update_lr(self.optimizer.lr().iter().map(|r| r * self.gamma).collect());
        }
    }

    fn reset_grad(&mut self, factory: &mut TensorFactory) {
        self.optimizer.reset_grad(factory);
    }
}

impl StepDecay {
    pub fn new(optimizer: Box<dyn Optimizer>, step_size: usize, gamma: Option<f64>) -> Self {
        Self {
            optimizer,
            step_size,
            gamma: gamma.unwrap_or(0.01),
        }
    }
}
