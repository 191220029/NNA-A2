use std::{hash::Hash, ops::Add};

use ndarray::{ArrayD, Axis};

use crate::op::op::{EWiseAdd, Op};

use super::tensor_factory::TensorFactory;

pub type TensorId = u32;

#[derive(Clone, Debug)]
pub struct Tensor {
    pub id: u32,
    pub grad: Option<ArrayD<f64>>,
    pub cached_data: Option<ArrayD<f64>>,
    pub op: Option<Op>,
    pub inputs: Vec<TensorId>,
    pub requires_grad: bool,
}

impl Default for Tensor {
    fn default() -> Self {
        Self {
            id: 0,
            grad: None,
            cached_data: None,
            op: None,
            inputs: vec![],
            requires_grad: true,
        }
    }
}

impl Tensor {
    pub fn is_leaf(&self) -> bool {
        self.op.is_none()
    }

    pub fn shape(&self) -> &[usize] {
        self.cached_data.as_ref().unwrap().shape()
    }
    pub fn from(value: ArrayD<f64>, factory: &mut TensorFactory) -> TensorId {
        let t = Self {
            id: 0,
            grad: None,
            cached_data: Some(value),
            op: None,
            inputs: vec![],
            requires_grad: true,
        };
        factory.insert_tensor(t)
    }
    
    pub fn sum_tensors(tensors: Vec<ArrayD<f64>>, shape: &[usize]) -> ArrayD<f64> {
        let mut iter = tensors.iter();
        let mut result = iter.next().unwrap().clone();
        while let Some(t) = iter.next() {
            result = result + t;
        }
        if result.shape() != shape {
            let axis = result.shape().len() - shape.len();
            result = result.sum_axis(Axis(axis)).to_shared().reshape(shape).to_owned();
        }
        result
    }
}
