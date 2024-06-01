use ndarray::{Array, ArrayD};

use crate::tensor::{tensor::TensorId, tensor_factory::TensorFactory};

fn init_he(in_features: usize, out_features: usize, factory: &mut TensorFactory) -> TensorId {
    let stddev = f64::sqrt(2. / in_features as f64);
    let mut rng = rand::thread_rng();
    let params = ArrayD::from_shape_fn(&[in_features, out_features], |(_, _)| {
        rng.gen();
    });
    return Tensor(numpy.random.randn(in_features, out_features).astype(dtype) * stddev, requires_grad=True)
}