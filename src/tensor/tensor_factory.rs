use std::collections::{HashMap, HashSet};

use ndarray::ArrayD;

use crate::op::op::Op;

use super::tensor::{Tensor, TensorId};

#[derive(Default)]
pub struct TensorFactory {
    tensor_map: HashMap<TensorId, Tensor>,
}

impl TensorFactory {
    pub fn insert_tensor(&mut self, mut tensor: Tensor) -> TensorId {
        match self.tensor_map.to_owned().keys().max() {
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
        let mut tensor = Tensor {
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
            ArrayD::zeros(node.shape())
        };

        compute_gradient_of_variables(node.id, out_grad, self)
    }
    pub fn realize_cached_data(&mut self, k: &TensorId) {
        let t = self.get(k).unwrap();

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
        ));
        if node.is_leaf() {
            continue;
        }
        for (i, grad) in node
            .op
            .as_ref()
            .unwrap()
            .gradient(node.grad.as_ref().unwrap(), &node)
            .into_iter()
            .enumerate()
        {
            let j = &node.inputs[i];
            if !node_to_output_grads_list.contains_key(j) {
                node_to_output_grads_list.insert(j.clone(), vec![]);
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

    use crate::tensor::tensor_factory::TensorFactory;

    use super::Tensor;

    #[test]
    fn test_auto_gradient() {
        let mut factory = TensorFactory::default();

        let a = Tensor::from(ArrayD::zeros(IxDyn(&[2, 2])), &mut factory);
        let b = Tensor::from(ArrayD::ones(IxDyn(&[2, 2])), &mut factory);

        let c = &a + &b;

        factory.backward(&c, None);

        println!("a={:?}", a);
        println!("b={:?}", b);
        println!("c={:?}", c);
    }
}
