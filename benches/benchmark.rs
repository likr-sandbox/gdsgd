use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use mdsgd::{
    classical_mds, double_centering, eigendecomposition, generate_graph,
    multi_source_shortest_path, pivot_mds,
};
use ndarray::*;
use ndarray_linalg::*;
use petgraph::prelude::*;
use rand::prelude::*;
use std::time::Duration;

fn create_graph(n: usize, p: f32) -> UnGraph<(), ()> {
    let mut rng = thread_rng();
    generate_graph(n, p, &mut rng)
}

fn create_matrix(graph: &UnGraph<(), ()>) -> Array2<f32> {
    let sources = graph.node_indices().collect::<Vec<_>>();
    let mut delta = multi_source_shortest_path(&graph, &sources);
    delta = delta.mapv_into(|v| v.powi(2));
    double_centering(&delta)
}

fn criterion_benchmark(c: &mut Criterion) {
    {
        let mut ed_group = c.benchmark_group("Eigendecomposition");
        for n in (100..=1000).step_by(100) {
            let p = 0.1;
            let graph = create_graph(n, p);
            let a = create_matrix(&graph);
            ed_group.bench_with_input(BenchmarkId::new("power_iteration", n), &a, |bench, a| {
                bench.iter(|| {
                    eigendecomposition(&a, 2);
                });
            });

            ed_group.bench_with_input(BenchmarkId::new("eig", n), &a, |bench, a| {
                bench.iter(|| {
                    a.eig().unwrap();
                });
            });

            ed_group.bench_with_input(BenchmarkId::new("eigh", n), &a, |bench, a| {
                bench.iter(|| {
                    a.eigh(UPLO::Upper).unwrap();
                });
            });
        }
    }

    {
        let mut mds_group = c.benchmark_group("MDS");
        for n in (100..=1000).step_by(100) {
            let p = 0.1;
            let graph = create_graph(n, p);
            mds_group.bench_with_input(
                BenchmarkId::new("Classical MDS", n),
                &graph,
                |bench, graph| {
                    bench.iter(|| {
                        classical_mds(&graph);
                    });
                },
            );

            mds_group.bench_with_input(BenchmarkId::new("Pivot MDS", n), &graph, |bench, graph| {
                let mut rng = thread_rng();
                bench.iter(|| {
                    pivot_mds(&graph, 10, &mut rng);
                });
            });
        }
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().measurement_time(Duration::new(5, 0));
    targets = criterion_benchmark
}
criterion_main!(benches);
