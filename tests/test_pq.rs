mod common;

use common::{Dataset, make_dataset, random_doubles};
use vq::rng::Mt19937_64;
use vq::{Error, ProductQuantizer, kmeans_plus_plus, l2_distance};

/// n points tightly scattered around `clusters` random centers.
fn make_clustered(n: usize, dim: usize, clusters: usize, seed: u64) -> Dataset {
    let centers = make_dataset(clusters, dim, seed, 10.0);
    let mut ds = make_dataset(n, dim, seed + 1, 0.01);
    for i in 0..n {
        for (value, center) in ds.row_mut(i).iter_mut().zip(centers.row(i % clusters)) {
            *value += center;
        }
    }
    ds
}

#[test]
fn kmeans_plus_plus_covers_clusters() {
    let (dim, clusters) = (4, 8);
    let ds = make_clustered(400, dim, clusters, 9);
    for seed in 0..10 {
        let mut rng = Mt19937_64::new(seed);
        let mut centroids = vec![0.0; clusters * dim];
        kmeans_plus_plus(&ds.data, dim, clusters, &mut centroids, &mut rng);
        // Points i and i % clusters share a cluster, so the first `clusters` points
        // are one representative per cluster. Each must have a centroid nearby.
        for j in 0..clusters {
            let covered =
                centroids.chunks_exact(dim).any(|c| l2_distance(c, ds.row(j)) < 0.01);
            assert!(covered, "cluster {j} not covered with seed {seed}");
        }
    }
}

#[test]
fn rejects_bad_params() {
    let ds = make_dataset(16, 8, 1, 1.0);
    let train = |dim, m, k| ProductQuantizer::train(&ds.data, dim, m, k, 10).map(|_| ());
    assert_eq!(train(8, 3, 4), Err(Error::InvalidSubspaces));
    assert_eq!(train(8, 2, 257), Err(Error::InvalidK));
    assert_eq!(train(8, 2, 32), Err(Error::TooFewVectors));
    assert_eq!(train(0, 2, 4), Err(Error::InvalidSubspaces));
    assert_eq!(train(8, 0, 4), Err(Error::InvalidSubspaces));
    assert_eq!(train(8, 2, 0), Err(Error::InvalidK));
    assert_eq!(
        ProductQuantizer::train(&ds.data[1..], 8, 2, 4, 10).map(|_| ()),
        Err(Error::DataLengthMismatch)
    );
}

#[test]
fn does_not_modify_input() {
    let ds = make_dataset(200, 8, 2, 1.0);
    let before = ds.data.clone();
    ProductQuantizer::train(&ds.data, 8, 2, 16, 10).unwrap();
    assert_eq!(ds.data, before);
}

#[test]
fn exact_when_k_equals_n() {
    let ds = make_dataset(16, 8, 3, 1.0);
    let pq = ProductQuantizer::train(&ds.data, 8, 4, 16, 10).unwrap();
    let mut out = vec![0.0; 8];
    for i in 0..ds.n {
        let v = ds.row(i);
        pq.decode(&pq.encode(v), &mut out);
        assert!(l2_distance(v, &out).abs() <= 1e-10);
    }
}

#[test]
fn clustered_reconstruction() {
    let dim = 16;
    let ds = make_clustered(500, dim, 8, 4);
    let pq = ProductQuantizer::train(&ds.data, dim, 4, 8, 25).unwrap();
    let mut out = vec![0.0; dim];
    for i in 0..ds.n {
        let v = ds.row(i);
        pq.decode(&pq.encode(v), &mut out);
        // Noise is 0.01 per coord, so squared error should be ~dim * 1e-4.
        assert!(l2_distance(v, &out) < 0.05);
    }
}

#[test]
fn adc_matches_decoded_distance() {
    let dim = 16;
    let ds = make_dataset(300, dim, 5, 1.0);
    let queries = make_dataset(5, dim, 6, 1.0);
    let pq = ProductQuantizer::train(&ds.data, dim, 4, 16, 15).unwrap();
    let codes = pq.encode_batch(&ds.data);
    let mut decoded = vec![0.0; dim];
    for qi in 0..queries.n {
        let q = queries.row(qi);
        let results = pq.search(q, &codes, 10);
        assert_eq!(results.len(), 10);
        assert!(results.windows(2).all(|pair| pair[0].1 <= pair[1].1));
        for &(id, dist) in &results {
            pq.decode(&codes[id * pq.m()..(id + 1) * pq.m()], &mut decoded);
            assert!((dist - l2_distance(q, &decoded)).abs() <= 1e-3);
        }
    }
}

#[test]
fn batch_encode_matches_single() {
    let ds = make_dataset(100, 8, 10, 1.0);
    let pq = ProductQuantizer::train(&ds.data, 8, 4, 16, 10).unwrap();
    let batch = pq.encode_batch(&ds.data);
    assert_eq!(batch.len(), ds.n * pq.m());
    for i in 0..ds.n {
        assert_eq!(pq.encode(ds.row(i)), batch[i * pq.m()..(i + 1) * pq.m()]);
    }
}

#[test]
fn self_query_is_top1() {
    let dim = 16;
    let ds = make_clustered(64, dim, 64, 7); // every point its own cluster
    let pq = ProductQuantizer::train(&ds.data, dim, 4, 64, 25).unwrap();
    let codes = pq.encode_batch(&ds.data);
    for i in 0..64 {
        assert_eq!(pq.search(ds.row(i), &codes, 1)[0].0, i);
    }
}

#[test]
fn double() {
    let data = random_doubles(64 * 4, 8);
    let pq = ProductQuantizer::train(&data, 4, 2, 8, 10).unwrap();
    assert_eq!(pq.encode(&data[..4]).len(), 2);
}
