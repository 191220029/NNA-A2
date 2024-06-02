use std::path::Path;

use torchlib::{dataset::data_set::RawDataSet, loss::{loss::Loss, mse::MSELoss}, module::{linear::Linear, Module}, optimizer::{optimizer::Optimizer, sgd::SGD}, tensor::tensor_factory::TensorFactory};

fn demo_1() {
    let dataset = RawDataSet::read_from_csv(&Path::new("../data/iris.csv")).discretization();
    let x = dataset.get_x();
    let y = dataset.get_y();

    let mut factory = TensorFactory::default();
    let mut model = Linear::new(x.shape()[1], y.shape()[1], None, &mut factory);
    let mut opt = SGD::new(model.parameters(), 0.1);
    let loss = MSELoss::new();
    for _ in 0..100 {
        opt.reset_grad(&mut factory);
        
        let predicted = model.forward(x.to_owned(), &mut factory);
        let l = loss.loss(predicted, y.clone(), &mut factory);

        factory.backward(&l, None);
        opt.step(&mut factory);

        println!("loss={}", factory.get(&l).unwrap().cached_data.clone().unwrap().to_string());
    }

}

fn main() {
    println!("Hello, world!");

    demo_1();
}
