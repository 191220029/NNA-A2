use ndarray::{ArrayD, IxDyn};
use rand::{
    distributions::{Distribution, Uniform},
    Rng,
};

use crate::tensor::{tensor::TensorId, tensor_factory::TensorFactory};

pub fn init_he(in_features: usize, out_features: usize, factory: &mut TensorFactory) -> TensorId {
    let stddev = f64::sqrt(2. / in_features as f64);
    let mut rng = rand::thread_rng();
    let weights = ArrayD::from_shape_fn(IxDyn(&[in_features, out_features]), |_| rng.gen());
    factory.new_tensor(weights, None)
}

//// 均匀分布版Xavier初始化
pub fn init_xavier(
    in_features: usize,
    out_features: usize,
    factory: &mut TensorFactory,
) -> TensorId {
    let limit = f64::sqrt(6. / (in_features + out_features) as f64);
    let uniform = Uniform::new(-limit, limit);
    let mut rng = rand::thread_rng();
    let weights = ArrayD::from_shape_fn(IxDyn(&[in_features, out_features]), |_| {
        uniform.sample(&mut rng)
    });
    factory.new_tensor(weights, None)
}
