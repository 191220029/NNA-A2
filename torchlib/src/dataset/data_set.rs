use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use ndarray::{ArrayD, IxDyn};

pub struct RawDataSet {
    data: ArrayD<String>,
    labels: Vec<String>,
}

impl RawDataSet {
    pub fn read_from_csv(csv_file: &Path) -> Self {
        let mut reader = BufReader::new(File::open(csv_file).unwrap());
        let mut buf = String::new();
        let mut data: Vec<String> = vec![];
        let mut i = 0;

        let mut labels = vec![];
        while let Ok(u) = reader.read_line(&mut buf) {
            if u == 0 {
                break;
            }
            let splits = buf.split(",");
            if i == 0 {
                labels = splits.map(|s| s.trim().to_owned()).collect();
                i += 1;
            } else {
                data.append(&mut splits.map(|s| s.trim().to_owned()).collect());
                i += 1;
            }
            buf.clear();
        }

        Self {
            data: ArrayD::from_shape_vec(IxDyn(&[i - 1, labels.len()]), data)
                .unwrap()
                .to_owned(),
            labels,
        }
    }

    pub fn discretization(self) -> DataSet {
        let mut discretized_data: Vec<f64> = vec![];
        self.data.columns().into_iter().for_each(|c| {
            let mut set: HashMap<String, f64> = HashMap::new();
            discretized_data.append(
                &mut c
                    .into_iter()
                    .map(|e| match e.parse::<f64>() {
                        Ok(e) => e,
                        Err(_) => match set.get(e) {
                            Some(e) => *e,
                            None => {
                                set.insert(e.clone(), set.len() as f64);
                                (set.len() - 1) as f64
                            }
                        },
                    })
                    .collect(),
            )
        });

        let discretized_data =
            ArrayD::from_shape_vec(self.data.t().shape(), discretized_data).unwrap();
        DataSet {
            data: discretized_data.t().to_owned(),
            labels: self.labels,
        }
    }
}

pub struct DataSet {
    data: ArrayD<f64>,
    labels: Vec<String>,
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

#[cfg(test)]
mod test_data_set {
    use std::path::PathBuf;

    use super::RawDataSet;

    #[test]
    fn test_data_set_iris() {
        let data_set = RawDataSet::read_from_csv(&PathBuf::from("../data/iris.csv"))
            .discretization()
            .normalize();
        assert_eq!(data_set.data.shape(), &[150, 5]);
        assert_eq!(data_set.get_x().shape(), &[150, 4]);
        assert_eq!(data_set.get_y().shape(), &[150, 1]);
    }
}
