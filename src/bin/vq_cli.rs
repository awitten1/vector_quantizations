use std::time::Instant;

use vq::rng::{Mt19937_64, Normal};
use vq::{ProductQuantizer, ScalarQuantizer, l2_distance, top_n};

const N: usize = 10000;
const DIM: usize = 64;
const QUERIES: usize = 100;
const TOP_N: usize = 10;

/// Formats like `printf("%g")`: six significant digits, no trailing zeros.
fn g(value: f64) -> String {
    const PRECISION: i32 = 6;
    if value == 0.0 || !value.is_finite() {
        return value.to_string().to_lowercase();
    }
    let trim = |s: &str| match s.contains('.') {
        true => s.trim_end_matches('0').trim_end_matches('.').to_owned(),
        false => s.to_owned(),
    };
    let scientific = format!("{:.*e}", (PRECISION - 1) as usize, value);
    let (mantissa, exponent) = scientific.split_once('e').unwrap();
    let exponent: i32 = exponent.parse().unwrap();
    if !(-4..PRECISION).contains(&exponent) {
        let sign = if exponent < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", trim(mantissa), exponent.abs())
    } else {
        trim(&format!("{:.*}", (PRECISION - 1 - exponent) as usize, value))
    }
}

/// The dataset and queries shared by both quantizers.
struct Workload {
    data: Vec<f32>,
    queries: Vec<f32>,
    exact_top: Vec<Vec<usize>>, // exact top-TOP_N per query
}

/// Prints compression, mean squared reconstruction error and recall@TOP_N
/// against the exact top-TOP_N for each query.
fn report(
    name: &str,
    train_s: f64,
    bytes_per_vector: usize,
    workload: &Workload,
    decode: impl Fn(usize, &mut [f32]),
    search: impl Fn(&[f32]) -> Vec<(usize, f32)>,
) {
    let mut mse = 0.0;
    let mut decoded = vec![0.0; DIM];
    for (i, vector) in workload.data.chunks_exact(DIM).enumerate() {
        decode(i, &mut decoded);
        mse += f64::from(l2_distance(vector, &decoded));
    }
    mse /= N as f64;

    let mut recall = 0.0;
    for (query, truth) in workload.queries.chunks_exact(DIM).zip(&workload.exact_top) {
        for (id, _) in search(query) {
            if truth.contains(&id) {
                recall += 1.0;
            }
        }
    }
    recall /= (QUERIES * TOP_N) as f64;

    println!("\n{name}");
    println!("  train: {} s", g(train_s));
    println!(
        "  bytes/vector: {bytes_per_vector} ({}x compression)",
        (DIM * size_of::<f32>()) / bytes_per_vector
    );
    println!("  mean squared reconstruction error: {} (data variance/vector: {DIM})", g(mse));
    println!("  recall@{TOP_N}: {}", g(recall));
}

fn main() -> Result<(), vq::Error> {
    const M: usize = 8;
    const K: usize = 256;
    const ITERS: usize = 25;

    println!("vector_quantizations v{}", vq::version());

    let mut rng = Mt19937_64::new(0);
    let mut dist = Normal::standard();
    let data: Vec<f32> = (0..N * DIM).map(|_| dist.sample(&mut rng)).collect();
    let queries: Vec<f32> = (0..QUERIES * DIM).map(|_| dist.sample(&mut rng)).collect();

    let exact_top = queries
        .chunks_exact(DIM)
        .map(|query| {
            let dists: Vec<f32> =
                data.chunks_exact(DIM).map(|vector| l2_distance(query, vector)).collect();
            top_n(&dists, TOP_N).into_iter().map(|(id, _)| id).collect()
        })
        .collect();
    let workload = Workload { data, queries, exact_top };
    let data = &workload.data;

    let t0 = Instant::now();
    let pq = ProductQuantizer::train(data, DIM, M, K, ITERS)?;
    let pq_train = t0.elapsed().as_secs_f64();
    let pq_codes = pq.encode_batch(data);
    report(
        "product quantization (m=8, k=256)",
        pq_train,
        M,
        &workload,
        |i, out| pq.decode(&pq_codes[i * M..(i + 1) * M], out),
        |query| pq.search(query, &pq_codes, TOP_N),
    );

    let t0 = Instant::now();
    let sq = ScalarQuantizer::train(data, DIM)?;
    let sq_train = t0.elapsed().as_secs_f64();
    let sq_codes = sq.encode_batch(data);
    report(
        "scalar quantization (8-bit, per-dimension range)",
        sq_train,
        DIM,
        &workload,
        |i, out| sq.decode(&sq_codes[i * DIM..(i + 1) * DIM], out),
        |query| sq.search(query, &sq_codes, TOP_N),
    );
    Ok(())
}
