use crate::{
    op::op::Reshape,
    tensor::{tensor::TensorId, tensor_factory::TensorFactory},
};

use super::Module;

pub struct Flatten {
    train: bool,
}

impl Module for Flatten {
    fn init(&mut self) {
        self.train = true;
    }

    fn forward(&mut self, x: TensorId, factory: &mut TensorFactory) -> TensorId {
        let shape = factory.get(&x).unwrap().shape();
        let batch_size = shape[0];
        let remains = if shape.len() > 1 {
            let mut t = 1;
            for y in &shape[1..] {
                t *= y;
            }
            t
        } else {
            1
        };

        factory.make_from_op(
            crate::op::op::Op::Reshape(Reshape {
                shape: vec![batch_size, remains],
            }),
            vec![x],
            None,
        )
    }

    fn eval(&mut self) {
        self.train = false;
    }

    fn train(&mut self) {
        self.train = true;
    }
}

impl Flatten {
    pub fn new() -> Self {
        Self { train: true }
    }
}

#[cfg(test)]
mod test_flatten {
    use ndarray::{ArrayD, IxDyn};

    use crate::{module::Module, tensor::tensor_factory::TensorFactory};

    use super::Flatten;

    #[test]
    fn test_flatten() {
        let mut flatten = Flatten::new();
        flatten.init();
        let factory = &mut TensorFactory::default();
        let t = factory.new_tensor(ArrayD::from_shape_vec(
            IxDyn(&[3, 4, 4]),
            vec![
                1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4.,
                1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4.,
                1., 2., 3., 4., 1., 2., 3., 4.,
            ],
        )
        .unwrap(), None);
        let t = &flatten.forward(
            t,
            factory,
        );
        assert_eq!("[[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],\n [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],\n [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]]",
        factory.get(t).unwrap().cached_data.as_ref().unwrap().to_string());
    }
}
