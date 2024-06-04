use std::f64::consts::PI;

use crate::optimizer::optimizer::Optimizer;

use super::scheduler::Scheduler;

pub struct CosineDecayWithWarmRestarts {
    optimizer: Box<dyn Optimizer>,
    t_i: usize,
    t_mult: usize,
    eta_min: usize,
    last_epoch: i32,
    t_cur: i32,
}

impl Scheduler for CosineDecayWithWarmRestarts {
    fn step(&mut self, epoch: usize, factory: &mut crate::tensor::tensor_factory::TensorFactory) {
        self.optimizer.step(factory);

        self.last_epoch = epoch as i32;
        if self.t_cur >= self.t_i as i32 {
            self.t_cur = self.t_cur - self.t_i as i32;
            self.t_i = self.t_i * self.t_mult;
        }

        self.optimizer.update_lr(
            self.optimizer
                .lr()
                .iter()
                .map(|r| {
                    self.eta_min as f64
                        + (r - self.eta_min as f64)
                            * (1. + f64::cos(PI * self.t_cur as f64 / self.t_i as f64))
                            / 2.
                })
                .collect(),
        );

        self.t_cur += 1;
    }

    fn reset_grad(&mut self, factory: &mut crate::tensor::tensor_factory::TensorFactory) {
        self.optimizer.reset_grad(factory);
    }
}

impl CosineDecayWithWarmRestarts {
    pub fn new(
        optimizer: Box<dyn Optimizer>,
        t_0: usize,
        t_mult: Option<usize>,
        eta_min: Option<usize>,
        last_epoch: Option<i32>,
    ) -> Self {
        Self {
            optimizer,
            t_i: t_0,
            t_mult: t_mult.unwrap_or(1),
            eta_min: eta_min.unwrap_or(0),
            last_epoch: last_epoch.unwrap_or(-1),
            t_cur: last_epoch.unwrap_or(-1),
        }
    }
}
