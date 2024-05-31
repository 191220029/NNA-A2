use std::{collections::HashMap, ops::Add};

use ndarray::ArrayD;

use crate::op::op::Op;

// pub type TensorId = u32;
// pub type TensorMap = HashMap<TensorId, Tensor>;
#[derive(Clone)]
pub struct Tensor<'a> {
    pub grad: Option<ArrayD<f64>>,
    pub cached_data: Option<ArrayD<f64>>,
    pub op: Option<Box<&'a dyn Op>>,
    pub inputs: Vec<Tensor<'a>>,
    pub requires_grad: bool,
}

impl Default for Tensor<'_> {
    fn default() -> Self {
        Self {
            grad: None,
            cached_data: None,
            op: None,
            inputs: vec![],
            requires_grad: true,
        }
    }
}

impl Tensor<'_> {
    fn is_leaf(&self) -> bool {
        self.op.is_none()
    }
    fn realize_cached_data(&mut self) -> ArrayD<f64> {
        if self.is_leaf() || self.cached_data.is_some() {
            return self.cached_data.clone().unwrap();
        }
        self.cached_data = Some(
            self.op.as_ref().unwrap().compute(
                self.inputs
                    .iter_mut()
                    .map(|i| i.realize_cached_data())
                    .collect(),
            ),
        );
        self.cached_data.clone().unwrap()
    }
}
