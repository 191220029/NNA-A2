use std::path::Path;

use torchlib::{
    dataset::byte_data_set::RawByteDataSet,
    loss::{loss::Loss, mse::MSELoss},
    module::{
        flatten::Flatten, linear::Linear, residual::Residual, sequential::Sequential, Module,
    },
    optimizer::{adam::Adam, optimizer::Optimizer},
    tensor::tensor_factory::TensorFactory,
};

fn main() {
    let dataset = RawByteDataSet::read_from_binary(
        &Path::new("../data/MNIST/raw/train-images-idx3-ubyte"),
        &Path::new("../data/MNIST/raw/train-labels-idx1-ubyte"),
    )
    .as_data_set();
    let factory = &mut TensorFactory::default();

    let x_shape = dataset.shape();
    let r = x_shape[1];
    let c = x_shape[2];

    let flatten = Flatten::new();
    let linear = Linear::new(r * c, r * c, None, factory);
    let residual = Residual::new(Box::new(linear));
    let linear_end = Linear::new(r * c, 1, None, factory);
    let mut model = Sequential::new(vec![
        Box::new(flatten),
        Box::new(residual),
        Box::new(linear_end),
    ]);

    let mut opt = Adam::new(model.parameters(), None, None, None, None, &factory);

    for i in 0..2 {
        let x = factory.new_tensor(dataset.get_x(), None);
        let loss = MSELoss::new();
        let y = model.forward(x, factory);
        let l = loss.loss(y, dataset.get_y(), factory);

        factory.backward(&l, None, None);
        opt.step(factory);
        println!("episode={i}");
    }
}
