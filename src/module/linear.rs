use ndarray::{ArrayD, IxDyn};

use crate::{
    op::op::MatrixMul,
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
        unimplemented!()
    }

    fn children(&self) -> Vec<Box<&dyn Module>> {
        unimplemented!()
    }

    fn forward(&mut self, x: TensorId, factory: &mut TensorFactory) -> TensorId {
        let x_out = factory.make_from_op(
            crate::op::op::Op::MatMul(MatrixMul {}),
            vec![x, self.weight],
            None,
        );
        if let Some(bias) = self.bias {}

        todo!()

        // X_out = X @ self.weight
        // if self.bias:
        //     t = self.bias.broadcast_to(X_out.shape)
        //     return X_out + t
        // return X_out
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
        Self {
            train: false,
            in_features,
            out_features,
            weight: init_he(in_features, out_features, facory),
            bias: if let Some(bias) = bias {
                if bias {
                    Some(facory.new_tensor(ArrayD::zeros(IxDyn(&[out_features])), None))
                } else {
                    None
                }
            } else {
                None
            },
        }
    }
}
