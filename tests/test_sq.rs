mod common;

use common::{Dataset, make_dataset, random_doubles};
use vq::{Error, ScalarQuantizer, l2_distance};

/// Per-dimension bucket width, computed independently of the quantizer.
fn steps(ds: &Dataset) -> Vec<f32> {
    (0..ds.dim)
        .map(|d| {
            let column = || (0..ds.n).map(|i| ds.row(i)[d]);
            let lo = column().fold(f32::INFINITY, f32::min);
            let hi = column().fold(f32::NEG_INFINITY, f32::max);
            (hi - lo) / 256.0
        })
        .collect()
}

#[test]
fn rejects_bad_params() {
    let ds = make_dataset(4, 4, 1, 1.0);
    let train = |data, dim| ScalarQuantizer::<f32>::train(data, dim).map(|_| ());
    assert_eq!(train(&ds.data[..0], 4), Err(Error::NoTrainingVectors));
    assert_eq!(train(&ds.data, 0), Err(Error::InvalidDim));
    assert_eq!(train(&ds.data[1..], 4), Err(Error::DataLengthMismatch));
}

#[test]
fn error_within_half_bucket() {
    let ds = make_dataset(500, 16, 2, 1.0);
    let sq = ScalarQuantizer::train(&ds.data, ds.dim).unwrap();
    let step = steps(&ds);
    let mut out = vec![0.0; ds.dim];
    for i in 0..ds.n {
        sq.decode(&sq.encode(ds.row(i)), &mut out);
        for d in 0..ds.dim {
            assert!((ds.row(i)[d] - out[d]).abs() <= step[d] / 2.0 + 1e-5);
        }
    }
}

#[test]
fn out_of_range_clamps() {
    let ds = make_dataset(100, 4, 3, 1.0);
    let sq = ScalarQuantizer::train(&ds.data, ds.dim).unwrap();
    assert!(sq.encode(&[-1e6; 4]).iter().all(|&c| c == 0));
    assert!(sq.encode(&[1e6; 4]).iter().all(|&c| c == 255));
}

#[test]
fn constant_dimension_is_exact() {
    let mut ds = make_dataset(50, 3, 4, 1.0);
    for i in 0..ds.n {
        ds.row_mut(i)[1] = 7.25;
    }
    let sq = ScalarQuantizer::train(&ds.data, ds.dim).unwrap();
    let mut out = vec![0.0; ds.dim];
    sq.decode(&sq.encode(ds.row(0)), &mut out);
    assert_eq!(out[1], 7.25);
}

#[test]
fn distances_match_decoded() {
    let ds = make_dataset(300, 16, 5, 1.0);
    let queries = make_dataset(5, 16, 6, 1.0);
    let sq = ScalarQuantizer::train(&ds.data, ds.dim).unwrap();
    let codes = sq.encode_batch(&ds.data);
    let (mut dists, mut decoded) = (vec![0.0; ds.n], vec![0.0; ds.dim]);
    for qi in 0..queries.n {
        let q = queries.row(qi);
        sq.distances(q, &codes, &mut dists);
        for i in 0..ds.n {
            sq.decode(&codes[i * ds.dim..(i + 1) * ds.dim], &mut decoded);
            assert!((dists[i] - l2_distance(q, &decoded)).abs() <= 1e-3);
        }
    }
}

#[test]
fn search() {
    let ds = make_dataset(200, 16, 7, 1.0);
    let sq = ScalarQuantizer::train(&ds.data, ds.dim).unwrap();
    let codes = sq.encode_batch(&ds.data);
    for i in 0..ds.n {
        let results = sq.search(ds.row(i), &codes, 10);
        assert_eq!(results.len(), 10);
        assert_eq!(results[0].0, i);
        assert!(results.windows(2).all(|pair| pair[0].1 <= pair[1].1));
    }
}

#[test]
fn batch_encode_matches_single() {
    let ds = make_dataset(100, 8, 8, 1.0);
    let sq = ScalarQuantizer::train(&ds.data, ds.dim).unwrap();
    let batch = sq.encode_batch(&ds.data);
    assert_eq!(batch.len(), ds.n * ds.dim);
    for i in 0..ds.n {
        assert_eq!(sq.encode(ds.row(i)), batch[i * ds.dim..(i + 1) * ds.dim]);
    }
}

#[test]
fn double() {
    let data = random_doubles(64 * 4, 9);
    let sq = ScalarQuantizer::train(&data, 4).unwrap();
    let mut out = vec![0.0; 4];
    sq.decode(&sq.encode(&data[..4]), &mut out);
    assert!(l2_distance(&data[..4], &out) < 0.01);
}
