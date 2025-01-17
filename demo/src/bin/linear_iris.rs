use std::path::Path;

use ndarray::{ArrayD, IxDyn};
use torchlib::{
    dataset::csv_data_set::RawCsvDataSet,
    loss::{loss::Loss, mse::MSELoss},
    module::{linear::Linear, relu::ReLU, sequential::Sequential, Module},
    optimizer::{optimizer::Optimizer, sgd::SGD},
    tensor::tensor_factory::TensorFactory,
};

fn main() {
    let dataset = RawCsvDataSet::read_from_csv(&Path::new("../data/iris.csv"))
        .discretization()
        .normalize();
    let x = dataset.get_x();
    let y = dataset.get_y();

    let mut factory = TensorFactory::default();
    let in_features = x.shape()[1];
    let linear = Linear::new(in_features, y.shape()[1], None, &mut factory);
    let relu = ReLU::new();
    let mut model = Sequential::new(vec![Box::new(linear), Box::new(relu)]);
    let mut opt = SGD::new(model.parameters(), 0.01);
    let loss = MSELoss::new();
    for episode in 0..100 {
        opt.reset_grad(&mut factory);

        let mut x = x.rows().into_iter();
        let mut y = y.rows().into_iter();

        while let Some(x) = x.next() {
            let x = ArrayD::from_shape_vec(IxDyn(x.shape()), x.to_vec()).unwrap();
            let y = y.next().unwrap();
            let y = ArrayD::from_shape_vec(IxDyn(y.shape()), y.to_vec()).unwrap();
            let x = factory.new_tensor(
                x.to_shared().reshape(IxDyn(&[1, in_features])).to_owned(),
                None,
            );
            let predicted = model.forward(x, &mut factory);
            let l = loss.loss(predicted, y.clone(), &mut factory);

            factory.backward(&l, None, None);
            opt.step(&mut factory);
        }

        if (episode + 1) % 10 == 0 {
            let x = factory.new_tensor(dataset.get_x(), None);
            let predicted = model.forward(x, &mut factory);
            let l = loss.loss(predicted, dataset.get_y(), &mut factory);

            println!(
                "episode {}, loss={}",
                episode + 1,
                factory
                    .get(&l)
                    .unwrap()
                    .cached_data
                    .clone()
                    .unwrap()
                    .to_string()
            );
        }
    }
}
