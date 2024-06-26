use std::path::Path;

use ndarray::{ArrayD, Axis, IxDyn, Slice};
use torchlib::{
    dataset::byte_data_set::RawByteDataSet,
    loss::{loss::Loss, mse::MSELoss},
    module::{
        flatten::Flatten, linear::Linear, relu::ReLU, residual::Residual, sequential::Sequential,
        Module,
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
    let relu = ReLU::new();
    let sequential = Sequential::new(vec![Box::new(linear), Box::new(relu)]);
    let residual = Residual::new(Box::new(sequential));
    let linear_end = Linear::new(r * c, 1, None, factory);
    let mut model = Sequential::new(vec![
        Box::new(flatten),
        Box::new(residual),
        Box::new(linear_end),
    ]);

    let mut opt = Adam::new(model.parameters(), None, None, None, None, &factory);

    let length = dataset.shape()[0] as isize / 6;
    let mut i = 0;
    let batch_size = 1000;
    let x = dataset.get_x();
    let y = dataset.get_y();
    while i < length {
        let x = x.slice_axis(Axis(0), Slice::new(0, Some(i + batch_size), 1));
        let y = y.slice_axis(Axis(0), Slice::new(0, Some(i + batch_size), 1));

        let x = factory.new_tensor(x.to_owned(), None);
        let loss = MSELoss::new();
        let y_hat = model.forward(x, factory);
        let l = loss.loss(y_hat, y.to_owned(), factory);
        factory.backward(&l, None, None);
        opt.step(factory);

        i += batch_size;
        println!("training {i}/{length}");
    }
}