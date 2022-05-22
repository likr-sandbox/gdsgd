use ndarray::prelude::*;
use petgraph::prelude::*;
use petgraph::{
    graph::{node_index, IndexType},
    EdgeType,
};
use rand::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::f32::INFINITY;

fn multi_source_shortest_path<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
    sources: &[NodeIndex<Ix>],
) -> Array2<f32> {
    let n = graph.node_count();
    let k = sources.len();
    let mut distance_matrix = Array::from_elem((n, k), INFINITY);
    for c in 0..k {
        let s = sources[c];
        let mut queue = VecDeque::new();
        queue.push_back(s);
        distance_matrix[[s.index(), c]] = 0.;
        while let Some(u) = queue.pop_front() {
            for v in graph.neighbors(u) {
                if distance_matrix[[v.index(), c]].is_infinite() {
                    queue.push_back(v);
                    distance_matrix[[v.index(), c]] = distance_matrix[[u.index(), c]] + 1.;
                }
            }
        }
    }
    distance_matrix
}

fn cos(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let ab = a.dot(b);
    let aa = a.dot(a);
    let bb = b.dot(b);
    ab / (aa * bb).sqrt()
}

fn double_centering(delta: &Array2<f32>) -> Array2<f32> {
    let n = delta.shape()[0];
    let k = delta.shape()[1];
    let sum_col = delta.mean_axis(Axis(1)).unwrap();
    let sum_row = delta.mean_axis(Axis(0)).unwrap();
    let sum_all = sum_col.mean().unwrap();
    let mut c = Array::zeros((n, k));
    for i in 0..n {
        for j in 0..k {
            c[[i, j]] = (sum_col[i] + sum_row[j] - delta[[i, j]] - sum_all) / 2.;
        }
    }
    c
}

fn power_iteration(a: &Array2<f32>) -> (f32, Array1<f32>) {
    let n = a.shape()[0];
    let mut x = Array1::from_elem(n, 1. / n as f32);
    let mut x_next;
    loop {
        x_next = a.dot(&x);
        if 1. - cos(&x_next, &x) < 1e-4 {
            break;
        }
        x_next /= x_next.dot(&x_next).sqrt();
        x = x_next;
    }
    let e = x_next.dot(&x_next) / x_next.dot(&x);
    x_next /= x_next.dot(&x_next).sqrt();
    (e, x_next)
}

fn generate_graph<R: Rng>(n: usize, p: f32, rng: &mut R) -> Graph<(), (), Undirected> {
    let mut graph = Graph::new_undirected();
    for _ in 0..n {
        graph.add_node(());
    }
    for u in 0..n {
        for v in 0..u {
            if rng.gen::<f32>() < p {
                graph.add_edge(node_index(u), node_index(v), ());
            }
        }
    }
    graph
}

fn eigendecomposition(a: &mut Array2<f32>, k: usize) -> (Array1<f32>, Array2<f32>) {
    let n = a.shape()[0];
    let mut e = Array1::zeros(k);
    let mut v = Array2::zeros((n, k));
    let (ei, vi) = power_iteration(a);
    e[0] = ei;
    v.slice_mut(s![.., 0]).assign(&vi);
    for i in 1..k {
        for r in 0..n {
            for c in 0..n {
                a[[r, c]] -= e[i - 1] * v[[r, i - 1]] * v[[c, i - 1]];
            }
        }
        let (ei, vi) = power_iteration(a);
        e[i] = ei;
        v.slice_mut(s![.., i]).assign(&vi);
    }
    (e, v)
}

fn classical_mds<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
) -> HashMap<NodeIndex<Ix>, (f32, f32)> {
    let sources = graph.node_indices().collect::<Vec<_>>();
    let mut delta = multi_source_shortest_path(&graph, &sources);
    delta = delta.mapv_into(|v| v.powi(2));
    let mut b = double_centering(&delta);
    let (e, v) = eigendecomposition(&mut b, 2);
    let xy = v.dot(&Array2::from_diag(&e.mapv(|v| v.sqrt())));
    let mut pos = HashMap::new();
    for (i, u) in graph.node_indices().enumerate() {
        pos.insert(u, (xy[[i, 0]], xy[[i, 1]]));
    }
    pos
}

fn pivot_mds<N, E, Ty: EdgeType, Ix: IndexType, R: Rng>(
    graph: &Graph<N, E, Ty, Ix>,
    k: usize,
    rng: &mut R,
) -> HashMap<NodeIndex<Ix>, (f32, f32)> {
    let mut sources = graph.node_indices().collect::<Vec<_>>();
    sources.shuffle(rng);
    let mut delta = multi_source_shortest_path(&graph, &sources[..k]);
    delta = delta.mapv_into(|v| v.powi(2));
    let c = double_centering(&delta);
    let mut ct_c = c.t().dot(&c);
    let (e, v) = eigendecomposition(&mut ct_c, 2);
    let xy = v.dot(&Array2::from_diag(&e.mapv(|v| v.sqrt())));
    let xy = c.dot(&xy);
    let mut pos = HashMap::new();
    for (i, u) in graph.node_indices().enumerate() {
        pos.insert(u, (xy[[i, 0]], xy[[i, 1]]));
    }
    pos
}

fn main() {
    let mut rng = rand::thread_rng();
    let graph = generate_graph(100, 0.1, &mut rng);
    let pos = pivot_mds(&graph, 10, &mut rng);
    println!("{:?}", pos);
    let pos = classical_mds(&graph);
    println!("{:?}", pos);
}
