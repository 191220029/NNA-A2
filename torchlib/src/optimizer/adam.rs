use ndarray::ArrayD;

use crate::{
    op::op::{from_matrix, into_matrix},
    tensor::{tensor::TensorId, tensor_factory::TensorFactory},
};
use peroxide::fuga::PowOps;

use super::optimizer::Optimizer;

pub struct Adam {
    params: Vec<TensorId>,
    lr: f64,
    beta_1: f64,
    beta_2: f64,
    eps: f64,
    m: Vec<ArrayD<f64>>,
    v: Vec<ArrayD<f64>>,
    t: usize,
}

impl Optimizer for Adam {
    fn step(&mut self, factory: &mut TensorFactory) {
        self.t += 1;
        let lr_t = self.lr * (1. - self.beta_2.powi(self.t as i32)).sqrt()
            / (1. - self.beta_1.powi(self.t as i32));
        for (i, param) in self.params.iter().enumerate() {
            self.m[i] = self.beta_1 * &self.m[i]
                + (1. - self.beta_1) * factory.get(param).unwrap().grad.as_ref().unwrap();

            let grad = into_matrix(factory.get(param).unwrap().grad.as_ref().unwrap().clone());
            self.v[i] = self.beta_2 * &self.v[i] + (1. - self.beta_2) * (from_matrix(grad.powi(2)));

            let m_hat = &self.m[i] / (1. - self.beta_1.powi(self.t as i32));
            let v_hat = &self.v[i] / (1. - self.beta_2.powi(self.t as i32));
            let v_hat = into_matrix(v_hat);

            factory.get_mut(param).unwrap().cached_data = Some(
                factory.get(param).unwrap().cached_data.as_ref().unwrap()
                    - lr_t * m_hat / (from_matrix(v_hat.sqrt()) + self.eps),
            );

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

impl Adam {
    pub fn new(
        params: Vec<TensorId>,
        lr: Option<f64>,
        beta_1: Option<f64>,
        beta_2: Option<f64>,
        eps: Option<f64>,
        factory: &TensorFactory,
    ) -> Self {
        let m: Vec<ArrayD<f64>> = params
            .iter()
            .map(|p| ArrayD::zeros(factory.get(p).unwrap().shape()))
            .collect();
        Self {
            params,
            lr: lr.unwrap_or(0.001),
            beta_1: beta_1.unwrap_or(0.9),
            beta_2: beta_2.unwrap_or(0.999),
            eps: eps.unwrap_or(1e-8),
            m: m.clone(),
            v: m,
            t: 0,
        }
    }
}
