use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use ndarray::{ArrayD, IxDyn};

use super::data_set::DataSet;

pub struct RawCsvDataSet {
    x: ArrayD<String>,
    y: ArrayD<String>,
    labels: Vec<String>,
}

impl RawCsvDataSet {
    pub fn read_from_csv(csv_file: &Path) -> Self {
        let mut reader = BufReader::new(File::open(csv_file).unwrap());
        let mut buf = String::new();
        let mut x: Vec<String> = vec![];
        let mut y: Vec<String> = vec![];
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
                let mut line: Vec<String> = splits.map(|s| s.trim().to_owned()).collect();
                let this_y = line.last().unwrap().to_string();
                line.remove(line.len() - 1);
                x.append(&mut line);
                y.push(this_y);
                i += 1;
            }
            buf.clear();
        }

        Self {
            x: ArrayD::from_shape_vec(IxDyn(&[i - 1, labels.len() - 1]), x)
                .unwrap()
                .to_owned(),
            y: ArrayD::from_shape_vec(IxDyn(&[i - 1, 1]), y)
                .unwrap()
                .to_owned(),
            labels,
        }
    }

    pub fn discretization(self) -> DataSet {
        let mut discretized_x: Vec<f64> = vec![];
        self.x.columns().into_iter().for_each(|c| {
            let mut set: HashMap<String, f64> = HashMap::new();
            discretized_x.append(
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
        let discretized_x = ArrayD::from_shape_vec(self.x.t().shape(), discretized_x).unwrap();

        let mut set: HashMap<String, f64> = HashMap::new();
        let discretized_y: Vec<f64> = self
            .y
            .into_iter()
            .map(|y| match y.parse::<f64>() {
                Ok(y) => y,
                Err(_) => match set.get(&y) {
                    Some(y) => *y,
                    None => {
                        set.insert(y.clone(), set.len() as f64);
                        (set.len() - 1) as f64
                    }
                },
            })
            .collect();

        DataSet {
            x: discretized_x.t().to_owned(),
            y: ArrayD::from_shape_vec(IxDyn(&[discretized_y.len(), 1]), discretized_y).unwrap(),
            labels: self.labels,
        }
    }
}

#[cfg(test)]
mod test_csv_data_set {
    use std::path::PathBuf;

    use super::RawCsvDataSet;

    #[test]
    fn test_data_set_iris() {
        let data_set = RawCsvDataSet::read_from_csv(&PathBuf::from("../data/iris.csv"))
            .discretization()
            .normalize();
        assert_eq!(data_set.get_x().shape(), &[150, 4]);
        assert_eq!(data_set.get_y().shape(), &[150, 1]);
    }
}
