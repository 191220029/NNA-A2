use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

use ndarray::{ArrayD, IxDyn};

use super::data_set::DataSet;

pub struct RawByteDataSet {
    x: ArrayD<u8>,
    y: Vec<u8>,
}

impl RawByteDataSet {
    pub fn read_from_binary(binary_data: &Path, binary_label: &Path) -> Self {
        Self {
            x: Self::read_data(binary_data),
            y: Self::read_label(binary_label),
        }
    }

    pub fn as_data_set(self) -> DataSet {
        DataSet {
            x: ArrayD::from_shape_vec(
                self.x.shape(),
                self.x
                    .clone()
                    .into_raw_vec()
                    .into_iter()
                    .map(|x| x as f64)
                    .collect(),
            )
            .unwrap(),
            y: ArrayD::from_shape_vec(
                IxDyn(&[self.y.len(), 1]),
                self.y.into_iter().map(|y| y as f64).collect(),
            )
            .unwrap(),
            labels: vec![],
        }
    }

    fn read_data(binary_data: &Path) -> ArrayD<u8> {
        let mut reader = BufReader::new(File::open(binary_data).unwrap());

        // magic number
        let _ = read_u32(&mut reader);

        let length = read_u32(&mut reader);

        let rows = read_u32(&mut reader);

        let columns = read_u32(&mut reader);

        let mut buf = vec![];
        reader.read_to_end(&mut buf).unwrap();
        assert_eq!(buf.len() as u32, length * rows * columns);

        ArrayD::from_shape_vec(
            IxDyn(&[length as usize, rows as usize, columns as usize]),
            buf,
        )
        .unwrap()
    }

    fn read_label(binary_data: &Path) -> Vec<u8> {
        let mut reader = BufReader::new(File::open(binary_data).unwrap());

        // magic number
        let _ = read_u32(&mut reader);

        let length = read_u32(&mut reader);

        let mut buf = vec![];
        let _ = reader.read_to_end(&mut buf);
        assert_eq!(buf.len() as u32, length);

        buf
    }
}

fn read_u32(reader: &mut impl Read) -> u32 {
    let mut buf = 0u32.to_le_bytes();
    reader.read_exact(&mut buf).unwrap();
    u32::from_be_bytes(buf)
}

#[cfg(test)]
mod test_byte_data_set {
    use std::path::PathBuf;

    use super::RawByteDataSet;

    #[test]
    fn test_byte_data_set() {
        let data_set = RawByteDataSet::read_from_binary(
            &PathBuf::from("../data/MNIST/raw/train-images-idx3-ubyte"),
            &PathBuf::from("../data/MNIST/raw/train-labels-idx1-ubyte"),
        )
        .as_data_set();

        assert_eq!(data_set.x.shape(), &[60000, 28, 28]);
        assert_eq!(data_set.y.len(), 60000);
    }
}
