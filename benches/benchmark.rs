use criterion::{criterion_group, criterion_main, Criterion};
use mdsgd::{double_centering, eigendecomposition, generate_graph, multi_source_shortest_path};
use ndarray::*;
use ndarray_linalg::*;
use rand::prelude::*;

fn create_matrix(n: usize, p: f32) -> Array2<f32> {
    let mut rng = thread_rng();
    let graph = generate_graph(n, p, &mut rng);
    let sources = graph.node_indices().collect::<Vec<_>>();
    let mut delta = multi_source_shortest_path(&graph, &sources);
    delta = delta.mapv_into(|v| v.powi(2));
    double_centering(&delta)
}

fn criterion_benchmark(c: &mut Criterion) {
    let n = 100;
    let p = 0.1;
    let a = create_matrix(n, p);
    c.bench_function("power iteration", |bench| {
        bench.iter(|| {
            eigendecomposition(&a, 2);
        });
    });

    c.bench_function("eig", |bench| {
        bench.iter(|| {
            a.eig().unwrap();
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
