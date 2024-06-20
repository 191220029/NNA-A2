use ndarray::ArrayD;

pub struct DataSet {
    pub(crate) x: ArrayD<f64>,
    pub(crate) y: ArrayD<f64>,
    pub labels: Vec<String>,
}

impl DataSet {
    pub fn normalize(mut self) -> Self {
        self.x.columns_mut().into_iter().for_each(|mut c| {
            let mean = c.mean().unwrap();
            let std = c.std(1.);
            c.iter_mut().for_each(|x| {
                *x = (*x - mean) / std;
            })
        });
        self
    }
    pub fn get_x(&self) -> ArrayD<f64> {
        self.x.clone()
    }
    pub fn get_y(&self) -> ArrayD<f64> {
        self.y.clone()
    }
    pub fn shape(&self) -> &[usize] {
        self.x.shape()
    }
}
