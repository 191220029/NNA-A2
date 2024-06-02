use ndarray::{ArrayD, IxDyn};
use ndarray_rand::{
    rand_distr::{Normal, Uniform},
    RandomExt,
};

use crate::tensor::{tensor::TensorId, tensor_factory::TensorFactory};

pub fn init_he(in_features: usize, out_features: usize, factory: &mut TensorFactory) -> TensorId {
    let stddev = f64::sqrt(2. / in_features as f64);
    let weights = ArrayD::random(
        IxDyn(&[in_features, out_features]),
        Normal::new(0., stddev).unwrap(),
    );
    let t = factory.new_tensor(weights, None);
    // eprintln!("init_He generates weight {t}: {:?}", factory.get(&t).unwrap().cached_data);
    return t;
}

//// 均匀分布版Xavier初始化
pub fn init_xavier(
    in_features: usize,
    out_features: usize,
    factory: &mut TensorFactory,
) -> TensorId {
    let limit = f64::sqrt(6. / (in_features + out_features) as f64);
    let uniform = Uniform::new(-limit, limit);
    let weights = ArrayD::random(IxDyn(&[in_features, out_features]), uniform);
    factory.new_tensor(weights, None)
}
