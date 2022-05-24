use mdsgd::{classical_mds, generate_graph, pivot_mds};
use rand::prelude::*;

fn main() {
    let mut rng = thread_rng();
    let graph = generate_graph(100, 0.1, &mut rng);
    let pos = pivot_mds(&graph, 10, &mut rng);
    println!("{:?}", pos);
    let pos = classical_mds(&graph);
    println!("{:?}", pos);
}
