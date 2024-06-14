use ndarray::{ArrayD, IxDyn};

use crate::{
    op::op::{BroadCast, EWiseAdd, MatrixMul},
    tensor::{tensor::TensorId, tensor_factory::TensorFactory},
};

use super::{initialize::init_he, Module};

pub struct Linear {
    train: bool,
    in_features: usize,
    out_features: usize,
    weight: TensorId,
    bias: Option<TensorId>,
}

impl Module for Linear {
    fn init(&mut self) {
        self.train = true;
    }

    fn parameters(&self) -> Vec<TensorId> {
        match self.bias {
            Some(b) => vec![self.weight, b],
            None => vec![self.weight],
        }
    }

    fn forward(&mut self, x: TensorId, factory: &mut TensorFactory) -> TensorId {
        let x_out = factory.make_from_op(
            crate::op::op::Op::MatMul(MatrixMul {}),
            vec![x, self.weight],
            None,
        );
        if let Some(bias) = self.bias {
            let t = factory.make_from_op(
                crate::op::op::Op::BCast(BroadCast {
                    shape: factory.get(&x_out).unwrap().shape().to_vec(),
                }),
                vec![bias],
                None,
            );
            factory.make_from_op(
                crate::op::op::Op::EWiseAdd(EWiseAdd {}),
                vec![x_out, t],
                None,
            )
        } else {
            x_out
        }
    }

    fn eval(&mut self) {
        self.train = false;
    }

    fn train(&mut self) {
        self.train = true;
    }
}

impl Linear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        bias: Option<bool>,
        facory: &mut TensorFactory,
    ) -> Self {
        let weight = init_he(in_features, out_features, facory);
        let b = facory.new_tensor(ArrayD::zeros(IxDyn(&[out_features])), None);
        Self {
            train: false,
            in_features,
            out_features,
            weight,
            bias: if let Some(bias) = bias {
                if bias {
                    Some(b)
                } else {
                    None
                }
            } else {
                None
            },
        }
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }
}
