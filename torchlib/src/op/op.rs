use std::f64::consts::LN_2;

use ndarray::{ArrayD, Axis, IxDyn};
use peroxide::fuga::{ExpLogOps, Matrix, MatrixProduct, PowOps, Shape::Row, Vector};

use crate::tensor::{tensor::Tensor, tensor_factory::TensorFactory};

#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    EWiseAdd(EWiseAdd),
    AddScalar(AddScalar),
    Sum(Summation),
    MatMul(MatrixMul),
    MulScalar(MulScalar),
    MulHadamard(MulHadamard),
    BCast(BroadCast),
    Neg(Negate),
    Power(PowerScalar),
    Reshape(Reshape),
    Max(MaxScalar),
    GetMax(GetMax),
    Exp(Exp),
    Log(Log),
    DivScalar(DivScalar),
    DivTensor(DivTensor),
}

impl Op {
    pub fn compute(&mut self, args: Vec<ArrayD<f64>>) -> ArrayD<f64> {
        match self {
            Op::EWiseAdd(e) => e.compute(args),
            Op::Sum(s) => s.compute(args),
            Op::MatMul(m) => m.compute(args),
            Op::MulScalar(m) => m.compute(args),
            Op::MulHadamard(m) => m.compute(args),
            Op::BCast(b) => b.compute(args),
            Op::Neg(n) => n.compute(args),
            Op::Power(p) => p.compute(args),
            Op::Reshape(r) => r.compute(args),
            Op::Max(m) => m.compute(args),
            Op::Exp(e) => e.compute(args),
            Op::AddScalar(a) => a.compute(args),
            Op::DivScalar(d) => d.compute(args),
            Op::DivTensor(d) => d.compute(args),
            Op::Log(l) => l.compute(args),
            Op::GetMax(g) => g.compute(args),
        }
    }
    pub fn gradient(
        &self,
        out_grad: &ArrayD<f64>,
        node: &Tensor,
        factory: &TensorFactory,
    ) -> Vec<ArrayD<f64>> {
        match self {
            Op::EWiseAdd(e) => e.gradient(out_grad, node, factory),
            Op::Sum(s) => s.gradient(out_grad, node, factory),
            Op::MatMul(m) => m.gradient(out_grad, node, factory),
            Op::MulScalar(m) => m.gradient(out_grad, node, factory),
            Op::MulHadamard(m) => m.gradient(out_grad, node, factory),
            Op::BCast(b) => b.gradient(out_grad, node, factory),
            Op::Neg(n) => n.gradient(out_grad, node, factory),
            Op::Power(p) => p.gradient(out_grad, node, factory),
            Op::Reshape(r) => r.gradient(out_grad, node, factory),
            Op::Max(m) => m.gradient(out_grad, node, factory),
            Op::Exp(e) => e.gradient(out_grad, node, factory),
            Op::AddScalar(a) => a.gradient(out_grad, node, factory),
            Op::DivScalar(d) => d.gradient(out_grad, node, factory),
            Op::DivTensor(d) => d.gradient(out_grad, node, factory),
            Op::Log(l) => l.gradient(out_grad, node, factory),
            Op::GetMax(g) => g.gradient(out_grad, node, factory),
        }
    }
}

trait OpTrait {
    fn compute(&mut self, args: Vec<ArrayD<f64>>) -> ArrayD<f64>;
    fn gradient(
        &self,
        out_grad: &ArrayD<f64>,
        node: &Tensor,
        factory: &TensorFactory,
    ) -> Vec<ArrayD<f64>>;
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct EWiseAdd {}
impl OpTrait for EWiseAdd {
    fn compute(&mut self, args: Vec<ArrayD<f64>>) -> ArrayD<f64> {
        if args[1].shape() != args[0].shape() {
            eprintln!("{:?} {:?}", args[1].shape(), args[0].shape());
            let t = args[1].broadcast(args[0].shape()).unwrap();
            &args[0] + &t
        } else {
            &args[0] + &args[1]
        }
    }
    fn gradient(&self, out_grad: &ArrayD<f64>, _: &Tensor, _: &TensorFactory) -> Vec<ArrayD<f64>> {
        vec![out_grad.to_owned(), out_grad.to_owned()]
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct AddScalar {
    pub scalar: f64,
}
impl OpTrait for AddScalar {
    fn compute(&mut self, args: Vec<ArrayD<f64>>) -> ArrayD<f64> {
        &args[0] + self.scalar
    }
    fn gradient(&self, out_grad: &ArrayD<f64>, _: &Tensor, _: &TensorFactory) -> Vec<ArrayD<f64>> {
        vec![out_grad.to_owned()]
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Negate {}
impl OpTrait for Negate {
    fn compute(&mut self, args: Vec<ArrayD<f64>>) -> ArrayD<f64> {
        -&args[0]
    }
    fn gradient(&self, out_grad: &ArrayD<f64>, _: &Tensor, _: &TensorFactory) -> Vec<ArrayD<f64>> {
        vec![-out_grad.to_owned()]
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct MatrixMul {}
impl OpTrait for MatrixMul {
    fn compute(&mut self, args: Vec<ArrayD<f64>>) -> ArrayD<f64> {
        from_matrix(into_matrix(args[0].to_owned()) * (into_matrix(args[1].to_owned())))
    }
    fn gradient(
        &self,
        out_grad: &ArrayD<f64>,
        node: &Tensor,
        factory: &TensorFactory,
    ) -> Vec<ArrayD<f64>> {
        let a = factory.get(&node.inputs[0]).unwrap();
        let b = factory.get(&node.inputs[1]).unwrap();
        let grad_a = from_matrix(
            into_matrix(out_grad.to_owned())
                * into_matrix(b.cached_data.clone().unwrap().to_owned()).t(),
        );
        let grad_b = from_matrix(
            into_matrix(a.cached_data.clone().unwrap().to_owned()).t()
                * into_matrix(out_grad.to_owned()),
        );
        return vec![grad_a, grad_b];
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct MulHadamard {}
impl OpTrait for MulHadamard {
    fn compute(&mut self, args: Vec<ArrayD<f64>>) -> ArrayD<f64> {
        from_matrix(into_matrix(args[0].to_owned()).hadamard(&into_matrix(args[1].to_owned())))
    }
    fn gradient(
        &self,
        out_grad: &ArrayD<f64>,
        node: &Tensor,
        factory: &TensorFactory,
    ) -> Vec<ArrayD<f64>> {
        return vec![
            from_matrix(
                into_matrix(out_grad.to_owned()).hadamard(&into_matrix(
                    factory
                        .get(&node.inputs[1])
                        .unwrap()
                        .cached_data
                        .as_ref()
                        .unwrap()
                        .to_owned(),
                )),
            ),
            from_matrix(
                into_matrix(out_grad.to_owned()).hadamard(&into_matrix(
                    factory
                        .get(&node.inputs[0])
                        .unwrap()
                        .cached_data
                        .as_ref()
                        .unwrap()
                        .to_owned(),
                )),
            ),
        ];
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct MulScalar {
    pub scalar: f64,
}
impl OpTrait for MulScalar {
    fn compute(&mut self, args: Vec<ArrayD<f64>>) -> ArrayD<f64> {
        from_matrix(into_matrix(args[0].to_owned()).mul_scalar(self.scalar))
    }
    fn gradient(&self, out_grad: &ArrayD<f64>, _: &Tensor, _: &TensorFactory) -> Vec<ArrayD<f64>> {
        return vec![from_matrix(
            into_matrix(out_grad.to_owned()).mul_scalar(self.scalar),
        )];
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Summation {
    pub axis: Option<usize>,
}
impl OpTrait for Summation {
    fn compute(&mut self, args: Vec<ArrayD<f64>>) -> ArrayD<f64> {
        match self.axis {
            Some(axis) => args[0].sum_axis(Axis(axis)),
            None => ArrayD::from_elem(IxDyn(&[1]), args[0].sum()),
        }
    }
    fn gradient(
        &self,
        out_grad: &ArrayD<f64>,
        node: &Tensor,
        factory: &TensorFactory,
    ) -> Vec<ArrayD<f64>> {
        let input_shape = factory.get(&node.inputs[0]).unwrap().shape();
        vec![out_grad.broadcast(input_shape).unwrap().to_owned()]
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct BroadCast {
    pub shape: Vec<usize>,
}
impl OpTrait for BroadCast {
    fn compute(&mut self, args: Vec<ArrayD<f64>>) -> ArrayD<f64> {
        args[0].broadcast(IxDyn(&self.shape)).unwrap().to_owned()
    }
    fn gradient(
        &self,
        out_grad: &ArrayD<f64>,
        node: &Tensor,
        factory: &TensorFactory,
    ) -> Vec<ArrayD<f64>> {
        let input_shape = factory.get(&node.inputs[0]).unwrap().shape();
        let mut grad = out_grad.to_owned();

        let y = self.shape.len() - input_shape.len();
        let mut t = vec![1usize].repeat(y);
        t.append(&mut input_shape.into_iter().map(|x| *x).collect());
        let matched_input_shape = t.as_slice();

        for (axis, dim) in matched_input_shape.iter().enumerate() {
            if *dim == 1 {
                grad = grad.sum_axis(Axis(axis))
            }
        }

        assert_eq!(grad.shape(), input_shape);

        return vec![grad];
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct PowerScalar {
    pub scalar: f64,
}
impl OpTrait for PowerScalar {
    fn compute(&mut self, args: Vec<ArrayD<f64>>) -> ArrayD<f64> {
        let m = into_matrix(args[0].to_owned()).powf(self.scalar);
        from_matrix(m)
    }
    fn gradient(
        &self,
        out_grad: &ArrayD<f64>,
        node: &Tensor,
        facotry: &TensorFactory,
    ) -> Vec<ArrayD<f64>> {
        let a = into_matrix(out_grad.to_owned());
        let a = a.mul_scalar(self.scalar);
        let t = into_matrix(
            facotry
                .get(&node.inputs[0])
                .unwrap()
                .cached_data
                .to_owned()
                .unwrap(),
        )
        .powf(self.scalar - 1.);

        vec![from_matrix(a.hadamard(&t))]
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Reshape {
    pub shape: Vec<usize>,
}
impl OpTrait for Reshape {
    fn compute(&mut self, args: Vec<ArrayD<f64>>) -> ArrayD<f64> {
        args[0].to_shared().reshape(IxDyn(&self.shape)).to_owned()
    }
    fn gradient(
        &self,
        out_grad: &ArrayD<f64>,
        node: &Tensor,
        factory: &TensorFactory,
    ) -> Vec<ArrayD<f64>> {
        vec![out_grad
            .to_shared()
            .reshape(factory.get(&node.inputs[0]).unwrap().shape())
            .to_owned()]
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct MaxScalar {
    pub scalar: f64,
}
impl OpTrait for MaxScalar {
    fn compute(&mut self, args: Vec<ArrayD<f64>>) -> ArrayD<f64> {
        let mut arg = args[0].clone();
        arg.iter_mut().for_each(|x| *x = x.max(self.scalar));
        arg
    }
    fn gradient(
        &self,
        out_grad: &ArrayD<f64>,
        node: &Tensor,
        _: &TensorFactory,
    ) -> Vec<ArrayD<f64>> {
        let grad = ArrayD::from_shape_vec(
            node.shape(),
            node.cached_data
                .as_ref()
                .unwrap()
                .iter()
                .map(|x| if *x > self.scalar { 1. } else { 0. })
                .collect(),
        )
        .unwrap();
        vec![from_matrix(
            into_matrix(out_grad.to_owned()).hadamard(&into_matrix(grad)),
        )]
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct GetMax {
    pub axis: usize,
}
impl OpTrait for GetMax {
    fn compute(&mut self, args: Vec<ArrayD<f64>>) -> ArrayD<f64> {
        let mut x = ArrayD::default(IxDyn(&[0]));
        args[0].axis_iter(Axis(self.axis)).for_each(|m| {
            if m.to_string().replace('[', " ").replace(']', " ").trim() > x.to_string().replace('[', " ").replace(']', " ").trim() {
                x = m.to_owned();
            }
        });
        let mut shape = args[0].shape().to_owned();
        shape[self.axis] = 1;
        x.to_shared().reshape(IxDyn(&shape)).to_owned()
    }
    fn gradient(
        &self,
        out_grad: &ArrayD<f64>,
        node: &Tensor,
        factory: &TensorFactory,
    ) -> Vec<ArrayD<f64>> {
        let mut grad = ArrayD::zeros(node.shape());
        let mut input_iter = factory
            .get(&node.inputs[0])
            .unwrap()
            .cached_data
            .as_ref()
            .unwrap()
            .axis_iter(Axis(self.axis));

        grad.axis_iter_mut(Axis(self.axis)).for_each(|mut x| {
            let y = input_iter.next().unwrap();
            if y == node.cached_data.as_ref().unwrap() {
                x.iter_mut().for_each(|x| *x = 1.);
            }
        });

        vec![from_matrix(
            into_matrix(out_grad.to_owned()).hadamard(&into_matrix(grad)),
        )]
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct DivScalar {
    pub scalar: f64,
}
impl OpTrait for DivScalar {
    fn compute(&mut self, args: Vec<ArrayD<f64>>) -> ArrayD<f64> {
        &args[0] / self.scalar
    }
    fn gradient(&self, out_grad: &ArrayD<f64>, _: &Tensor, _: &TensorFactory) -> Vec<ArrayD<f64>> {
        vec![out_grad / self.scalar]
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct DivTensor {
    scalar: f64,
}
impl OpTrait for DivTensor {
    fn compute(&mut self, args: Vec<ArrayD<f64>>) -> ArrayD<f64> {
        self.scalar = args[1].sum();
        &args[0] / self.scalar
    }
    fn gradient(
        &self,
        out_grad: &ArrayD<f64>,
        node: &Tensor,
        factory: &TensorFactory,
    ) -> Vec<ArrayD<f64>> {
        vec![
            out_grad / self.scalar,
            ArrayD::from_elem(
                IxDyn(&[1]),
                -(from_matrix(
                    into_matrix(out_grad.to_owned()).hadamard(&into_matrix(
                        factory
                            .get(&node.inputs[0])
                            .unwrap()
                            .cached_data
                            .as_ref()
                            .unwrap()
                            .to_owned(),
                    )),
                ))
                .sum()
                    / self.scalar.powi(2),
            ),
        ]
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Exp {}
impl OpTrait for Exp {
    fn compute(&mut self, args: Vec<ArrayD<f64>>) -> ArrayD<f64> {
        from_matrix(into_matrix(args[0].to_owned()).exp())
    }
    fn gradient(
        &self,
        out_grad: &ArrayD<f64>,
        node: &Tensor,
        factory: &TensorFactory,
    ) -> Vec<ArrayD<f64>> {
        vec![from_matrix(
            into_matrix(out_grad.to_owned()).hadamard(
                &into_matrix(
                    factory
                        .get(&node.inputs[0])
                        .unwrap()
                        .cached_data
                        .as_ref()
                        .unwrap()
                        .to_owned(),
                )
                .exp(),
            ),
        )]
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Log {}
impl OpTrait for Log {
    fn compute(&mut self, args: Vec<ArrayD<f64>>) -> ArrayD<f64> {
        from_matrix(into_matrix(args[0].to_owned()).log2())
    }
    fn gradient(
        &self,
        out_grad: &ArrayD<f64>,
        node: &Tensor,
        factory: &TensorFactory,
    ) -> Vec<ArrayD<f64>> {
        vec![from_matrix(
            into_matrix(out_grad.to_owned()).hadamard(
                &into_matrix(
                    factory
                        .get(&node.inputs[0])
                        .unwrap()
                        .cached_data
                        .as_ref()
                        .unwrap()
                        .to_owned(),
                )
                .mul_scalar(LN_2)
                .powi(-1),
            ),
        )]
    }
}

pub(crate) fn into_matrix(array: ArrayD<f64>) -> Matrix {
    let shape = array.shape().to_owned();
    match shape.get(1) {
        Some(x) => Matrix {
            data: array.into_raw_vec(),
            row: shape[0],
            col: *x,
            shape: Row,
        },
        None => Matrix {
            data: array.into_raw_vec(),
            row: 1,
            col: shape[0],
            shape: Row,
        },
    }
}

pub(crate) fn from_matrix(matrix: Matrix) -> ArrayD<f64> {
    ArrayD::from_shape_vec(IxDyn(&[matrix.row, matrix.col]), matrix.data).unwrap()
}
