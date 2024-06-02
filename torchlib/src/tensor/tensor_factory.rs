use std::collections::{HashMap, HashSet};

use ndarray::ArrayD;

use crate::op::op::Op;

use super::tensor::{Tensor, TensorId};

#[derive(Default)]
pub struct TensorFactory {
    tensor_map: HashMap<TensorId, Tensor>,
}

impl TensorFactory {
    pub fn new_tensor(&mut self, value: ArrayD<f64>, requires_grad: Option<bool>) -> TensorId {
        let t = Tensor {
            id: 0,
            grad: None,
            cached_data: Some(value),
            op: None,
            inputs: vec![],
            requires_grad: if let Some(rg) = requires_grad {
                rg
            } else {
                true
            },
        };
        self.insert_tensor(t)
    }

    pub fn make_from_op(
        &mut self,
        op: Op,
        inputs: Vec<TensorId>,
        requires_grad: Option<bool>,
    ) -> TensorId {
        let requires_grad = match requires_grad {
            Some(b) => b,
            None => {
                let mut flag = false;
                for i in &inputs {
                    flag |= self.get(i).unwrap().requires_grad;
                }
                flag
            }
        };
        let tensor = Tensor {
            id: 0,
            grad: None,
            cached_data: None,
            op: Some(op),
            inputs,
            requires_grad,
        };
        let t = self.insert_tensor(tensor);
        self.realize_cached_data(&t);
        t
    }
    pub fn get(&self, k: &TensorId) -> Option<&Tensor> {
        self.tensor_map.get(k)
    }
    pub fn get_mut(&mut self, k: &TensorId) -> Option<&mut Tensor> {
        self.tensor_map.get_mut(k)
    }
    pub fn remove(&mut self, k: &TensorId) -> Option<Tensor> {
        self.tensor_map.remove(k)
    }
    pub fn backward(&mut self, k: &TensorId, out_grad: Option<ArrayD<f64>>) {
        let node = self.get(k).unwrap();
        let out_grad = if let Some(out_grad) = out_grad {
            out_grad
        } else {
            ArrayD::ones(node.shape())
        };

        compute_gradient_of_variables(node.id, out_grad, self)
    }

    pub fn realize_cached_data(&mut self, k: &TensorId) {
        let t = self.get(k).unwrap();
        if t.is_leaf() || t.cached_data.is_some() {
            return;
        }

        let r = Some(
            t.op.clone().unwrap().compute(
                t.inputs
                    .clone()
                    .iter()
                    .map(|x| {
                        self.realize_cached_data(&x);
                        self.get(&x).unwrap().cached_data.clone().unwrap()
                    })
                    .collect(),
            ),
        );
        self.get_mut(k).unwrap().cached_data = r;
    }

    fn insert_tensor(&mut self, mut tensor: Tensor) -> TensorId {
        match self.tensor_map.keys().max().cloned() {
            Some(x) => {
                tensor.id = x + 1;
                self.tensor_map.insert(x + 1, tensor);
                x + 1
            }
            None => {
                tensor.id = 0;
                self.tensor_map.insert(0, tensor);
                0
            }
        }
    }
}

fn compute_gradient_of_variables(
    tensor_id: TensorId,
    out_grad: ArrayD<f64>,
    factory: &mut TensorFactory,
) {
    let mut node_to_output_grads_list: HashMap<TensorId, Vec<ArrayD<f64>>> = HashMap::new();
    node_to_output_grads_list.insert(tensor_id, vec![out_grad]);
    let mut reverse_topo_order = find_topo_sort(vec![tensor_id], &factory);
    reverse_topo_order.reverse();

    for id in reverse_topo_order {
        let node = factory.get_mut(&id).unwrap();

        node.grad = Some(Tensor::sum_tensors(
            node_to_output_grads_list.get(&node.id).unwrap().to_owned(),
            node.shape(),
        ));

        if node.is_leaf() {
            continue;
        }

        let node = factory.get(&id).unwrap();
        for (i, grad) in node
            .op
            .as_ref()
            .unwrap()
            .gradient(node.grad.as_ref().unwrap(), &node, &factory)
            .into_iter()
            .enumerate()
        {
            let j = &node.inputs[i];
            if !node_to_output_grads_list.contains_key(j) {
                node_to_output_grads_list.insert(*j, vec![]);
            }
            node_to_output_grads_list.get_mut(j).unwrap().push(grad);
        }
    }
}

fn find_topo_sort(tensors: Vec<TensorId>, factory: &TensorFactory) -> Vec<TensorId> {
    let mut visited = HashSet::new();
    let mut topo_order = vec![];

    fn dfs(
        visited: &mut HashSet<TensorId>,
        topo_order: &mut Vec<TensorId>,
        t: TensorId,
        factory: &TensorFactory,
    ) {
        if visited.contains(&t) {
            return;
        }
        visited.insert(t);
        for inp in &factory.get(&t).unwrap().inputs {
            dfs(visited, topo_order, *inp, factory);
        }
        topo_order.push(t.clone())
    }

    for t in tensors {
        dfs(&mut visited, &mut topo_order, t, factory);
    }
    return topo_order;
}

#[cfg(test)]
mod test_tensor {
    use ndarray::{ArrayD, IxDyn};

    use crate::{
        op::op::{BroadCast, Op},
        tensor::tensor_factory::TensorFactory,
    };

    #[test]
    fn test_auto_gradient() {
        let mut factory = TensorFactory::default();

        let a = factory.new_tensor(ArrayD::zeros(IxDyn(&[2, 1])), None);
        let b = factory.new_tensor(ArrayD::ones(IxDyn(&[1, 2])), None);

        let c = factory.make_from_op(Op::EWiseAdd(crate::op::op::EWiseAdd {}), vec![a, b], None);
        let d = factory.make_from_op(
            Op::Sum(crate::op::op::Summation { axis: None }),
            vec![c],
            None,
        );

        factory.backward(&d, None);

        assert_eq!(
            factory
                .get(&a)
                .unwrap()
                .grad
                .to_owned()
                .unwrap()
                .to_string(),
            "[[2],\n [2]]".to_string()
        );
        assert_eq!(
            factory
                .get(&b)
                .unwrap()
                .grad
                .to_owned()
                .unwrap()
                .to_string(),
            "[[2, 2]]".to_string()
        );
        assert_eq!(
            factory
                .get(&c)
                .unwrap()
                .grad
                .to_owned()
                .unwrap()
                .to_string(),
            "[[1, 1],\n [1, 1]]".to_string()
        );
        assert_eq!(
            factory
                .get(&d)
                .unwrap()
                .grad
                .to_owned()
                .unwrap()
                .to_string(),
            "[1]".to_string()
        );
        assert_eq!(
            factory
                .get(&d)
                .unwrap()
                .cached_data
                .to_owned()
                .unwrap()
                .to_string(),
            "[4]".to_string()
        );
    }

    #[test]
    fn test_auto_gradient_matmul() {
        let mut factory = TensorFactory::default();

        let a = factory.new_tensor(ArrayD::ones(IxDyn(&[5, 1])), None);
        let b = factory.new_tensor(ArrayD::ones(IxDyn(&[1, 5])), None);

        let c = factory.make_from_op(Op::MatMul(crate::op::op::MatrixMul {}), vec![a, b], None);

        let d = factory.make_from_op(
            Op::BCast(BroadCast {
                shape: vec![5, 5, 5, 5, 5],
            }),
            vec![c],
            None,
        );
        let e = factory.make_from_op(
            Op::Sum(crate::op::op::Summation { axis: None }),
            vec![d],
            None,
        );
        factory.backward(&e, None);

        assert_eq!(
            factory
                .get(&a)
                .unwrap()
                .grad
                .to_owned()
                .unwrap()
                .to_string(),
            "[[625],\n [625],\n [625],\n [625],\n [625]]".to_string()
        );
        assert_eq!(
            factory
                .get(&b)
                .unwrap()
                .grad
                .to_owned()
                .unwrap()
                .to_string(),
            "[[625, 625, 625, 625, 625]]".to_string()
        );
        assert_eq!(
            factory
                .get(&c)
                .unwrap()
                .grad
                .to_owned()
                .unwrap()
                .to_string(),
            "[[125, 125, 125, 125, 125],\n [125, 125, 125, 125, 125],\n [125, 125, 125, 125, 125],\n [125, 125, 125, 125, 125],\n [125, 125, 125, 125, 125]]".to_string()
        );
        assert_eq!(
            "[[[[[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]]],\n\n\n  [[[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]]],\n\n\n  [[[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]]],\n\n\n  [[[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]]],\n\n\n  [[[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]]]],\n\n\n\n [[[[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]]],\n\n\n  [[[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]]],\n\n\n  [[[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]]],\n\n\n  [[[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]]],\n\n\n  [[[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]]]],\n\n\n\n [[[[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]]],\n\n\n  [[[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]]],\n\n\n  [[[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]]],\n\n\n  [[[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]]],\n\n\n  [[[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]]]],\n\n\n\n [[[[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]]],\n\n\n  [[[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]]],\n\n\n  [[[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]]],\n\n\n  [[[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]]],\n\n\n  [[[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]]]],\n\n\n\n [[[[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]]],\n\n\n  [[[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]]],\n\n\n  [[[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]]],\n\n\n  [[[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]]],\n\n\n  [[[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]],\n\n   [[1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1],\n    [1, 1, 1, 1, 1]]]]]",
            factory
                .get(&d)
                .unwrap()
                .grad
                .to_owned()
                .unwrap()
                .to_string()
        );
        assert_eq!(
            factory
                .get(&e)
                .unwrap()
                .grad
                .to_owned()
                .unwrap()
                .to_string(),
            "[1]".to_string()
        );
        assert_eq!(
            factory
                .get(&e)
                .unwrap()
                .cached_data
                .to_owned()
                .unwrap()
                .to_string(),
            "[3125]".to_string()
        );
    }
}
