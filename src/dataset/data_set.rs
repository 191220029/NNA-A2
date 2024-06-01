use std::{
    collections::HashMap, fs::File, io::{BufRead, BufReader}, path::Path
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
            discretized_data.append(&mut c.into_iter().map(|e| {
                match e.parse::<f64>() {
                    Ok(e) => e,
                    Err(_) => {
                        match set.get(e) {
                            Some(e) => *e,
                            None => {
                                set.insert(e.clone(), set.len() as f64);
                                (set.len() - 1) as f64
                            },
                        }
                    },
                }
            }).collect())
        });

        let discretized_data = ArrayD::from_shape_vec(self.data.t().shape(), discretized_data).unwrap();
        DataSet { data: discretized_data.t().to_owned(), labels: self.labels }
    }

}

pub struct DataSet {
    data: ArrayD<f64>,
    labels: Vec<String>,
}

impl DataSet {
    pub fn get_x() -> ArrayD<f64> {
        unimplemented!()
    }
    pub fn get_y() -> ArrayD<f64> {
        unimplemented!()
    }
}

#[cfg(test)]
mod test_data_set {
    use std::path::PathBuf;

    use super::RawDataSet;

    #[test]
    fn test_data_set_iris() {
        let data_set = RawDataSet::read_from_csv(&PathBuf::from("data/iris.csv")).discretization();
        println!("{}", data_set.data);
    }
}