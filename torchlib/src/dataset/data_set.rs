use ndarray::{ArrayD, IxDyn};

pub struct DataSet {
    pub(crate) data: ArrayD<f64>,
    pub(crate) labels: Vec<String>,
}

impl DataSet {
    pub fn normalize(mut self) -> Self {
        self.data.columns_mut().into_iter().for_each(|mut c| {
            let mean = c.mean().unwrap();
            let std = c.std(1.);
            c.iter_mut().for_each(|x| {
                *x = (*x - mean) / std;
            })
        });
        self
    }
    pub fn get_x(&self) -> ArrayD<f64> {
        let it = self.data.columns().into_iter();
        let mut i = 0;
        let x: Vec<f64> = it
            .filter(|_| {
                i += 1;
                i < self.labels.len()
            })
            .map(|c| c.to_vec())
            .collect::<Vec<Vec<f64>>>()
            .into_iter()
            .flatten()
            .collect();
        ArrayD::from_shape_vec(IxDyn(&[self.shape()[0], self.shape()[1] - 1]), x).unwrap()
    }
    pub fn get_y(&self) -> ArrayD<f64> {
        let it = self.data.columns().into_iter();
        let mut i = 0;
        let x: Vec<f64> = it
            .filter(|_| {
                i += 1;
                i == self.labels.len()
            })
            .map(|c| c.to_vec())
            .collect::<Vec<Vec<f64>>>()
            .into_iter()
            .flatten()
            .collect();
        ArrayD::from_shape_vec(IxDyn(&[self.shape()[0], 1]), x).unwrap()
    }
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }
}
