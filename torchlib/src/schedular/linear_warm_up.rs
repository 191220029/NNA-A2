use crate::optimizer::optimizer::Optimizer;

use super::scheduler::Scheduler;

pub struct LinearWarmUp {
    optimizer: Box<dyn Optimizer>,
    warmup_epochs: usize,
    base_lr: f64,
    max_lr: f64,
}

impl Scheduler for LinearWarmUp {
    fn step(&mut self, epoch: usize, factory: &mut crate::tensor::tensor_factory::TensorFactory) {
        self.optimizer.step(factory);
        self.optimizer.update_lr(if epoch < self.warmup_epochs {
            vec![self.base_lr + (self.max_lr - self.base_lr) * (epoch / self.warmup_epochs) as f64]
                .repeat(self.optimizer.lr().len())
        } else {
            vec![self.max_lr].repeat(self.optimizer.lr().len())
        });
    }

    fn reset_grad(&mut self, factory: &mut crate::tensor::tensor_factory::TensorFactory) {
        self.optimizer.reset_grad(factory);
    }
}

impl LinearWarmUp {
    pub fn new(
        optimizer: Box<dyn Optimizer>,
        warmup_epochs: Option<usize>,
        base_lr: Option<f64>,
        max_lr: Option<f64>,
    ) -> Self {
        Self {
            optimizer,
            warmup_epochs: warmup_epochs.unwrap_or(5),
            base_lr: base_lr.unwrap_or(0.01),
            max_lr: max_lr.unwrap_or(0.1),
        }
    }
}
