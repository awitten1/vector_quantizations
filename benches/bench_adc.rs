//! Distance computation throughput: exact L2 over raw float vectors vs ADC
//! (asymmetric distance computation) over PQ codes vs scalar-quantized (SQ8) codes.
//!
//! Each scan benchmark computes the distance from one query to every database
//! vector and writes it to an output array, so all sides do the same job:
//!   Exact: dim multiply-adds per vector, reading dim * 4 bytes per vector.
//!   ADC:   m table lookups + adds per vector, reading m bytes per vector.
//!   SQ:    dim decodes + multiply-adds per vector, reading dim bytes per vector.
//! m = dim / 8 (dsub = 8), k = 256, so PQ codes are dim / 8 bytes.
//!
//! Usage: cargo bench -- [SUBSTRING] [--counters] [--min-time=SECONDS]
//!   SUBSTRING           only run benchmarks whose name contains it
//!   --counters          also report items/s, bytes/s and bytes per vector
//!   --min-time=SECONDS  minimum measured time per benchmark (default 0.5)

use std::collections::HashMap;
use std::hint::black_box;
use std::slice;
use std::time::{Duration, Instant};

use vq::rng::{self, Mt19937_64, Normal};
use vq::{ProductQuantizer, ScalarQuantizer, l2_distance};

const DSUB: usize = 8;
const K: usize = 256;
const TRAIN_N: usize = 2000; // codebook quality doesn't affect speed
const TRAIN_ITERS: usize = 3;
const DIMS: [usize; 3] = [64, 128, 256];
const SIZES: [usize; 2] = [10_000, 1_000_000];
const SINGLE_N: usize = 10_000; // single-distance benchmarks reuse this fixture

struct Fixture {
    dim: usize,
    n: usize,
    m: usize,
    data: Vec<f32>,    // n * dim raw vectors
    query: Vec<f32>,   // dim
    codes: Vec<u8>,    // n * m
    pq: ProductQuantizer<f32>,
    table: Vec<f32>,   // m * k, for the query
    sq: ScalarQuantizer<f32>,
    sq_codes: Vec<u8>, // n * dim
}

fn random_floats(count: usize, seed: u64) -> Vec<f32> {
    let mut rng = Mt19937_64::new(seed);
    let mut normal = Normal::standard();
    (0..count).map(|_| normal.sample(&mut rng)).collect()
}

impl Fixture {
    fn new(dim: usize, n: usize) -> Self {
        let m = dim / DSUB;
        let data = random_floats(n * dim, 1);
        let query = random_floats(dim, 2);
        let train = &data[..TRAIN_N * dim];
        let pq = ProductQuantizer::train(train, dim, m, K, TRAIN_ITERS).unwrap();
        let sq = ScalarQuantizer::train(train, dim).unwrap();
        // Random codes instead of encoding: scan cost depends only on the code
        // layout, and encoding n vectors would dominate setup time.
        let mut rng = Mt19937_64::new(3);
        let mut random_bytes = |count: usize| -> Vec<u8> {
            (0..count).map(|_| rng::uniform_below(&mut rng, K as u64) as u8).collect()
        };
        let codes = random_bytes(n * m);
        let sq_codes = random_bytes(n * dim);
        let table = pq.distance_table(&query);
        Self { dim, n, m, data, query, codes, pq, table, sq, sq_codes }
    }
}

struct Measurement {
    iterations: u64,
    elapsed: Duration,
    bytes_per_vector: Option<usize>, // set for scans, enables --counters
}

/// Runs `body` in growing batches until a batch lasts at least `min_time`.
fn measure(min_time: Duration, mut body: impl FnMut()) -> (u64, Duration) {
    let mut iterations = 1u64;
    loop {
        let start = Instant::now();
        for _ in 0..iterations {
            body();
        }
        let elapsed = start.elapsed();
        if elapsed >= min_time {
            return (iterations, elapsed);
        }
        let target = min_time.as_secs_f64() * 1.4 / elapsed.as_secs_f64().max(1e-9);
        iterations = (iterations as f64 * target.clamp(2.0, 10.0)).ceil() as u64;
    }
}

type Benchmark = fn(&Fixture, Duration) -> Measurement;

/// Baseline: exact squared L2 from the query to every raw vector.
fn exact_l2(f: &Fixture, min_time: Duration) -> Measurement {
    let mut out = vec![0.0; f.n];
    let (iterations, elapsed) = measure(min_time, || {
        for (slot, vector) in out.iter_mut().zip(f.data.chunks_exact(f.dim)) {
            *slot = l2_distance(&f.query, vector);
        }
        black_box(out.as_mut_ptr());
    });
    Measurement { iterations, elapsed, bytes_per_vector: Some(f.dim * size_of::<f32>()) }
}

/// ADC scan only: lookup table already built for this query.
fn adc_scan(f: &Fixture, min_time: Duration) -> Measurement {
    let mut out = vec![0.0; f.n];
    let (iterations, elapsed) = measure(min_time, || {
        f.pq.adc_distances(&f.table, &f.codes, &mut out);
        black_box(out.as_mut_ptr());
    });
    Measurement { iterations, elapsed, bytes_per_vector: Some(f.m) }
}

/// ADC end to end per query: build the m * k table, then scan.
fn adc_table_plus_scan(f: &Fixture, min_time: Duration) -> Measurement {
    let mut out = vec![0.0; f.n];
    let (iterations, elapsed) = measure(min_time, || {
        let table = f.pq.distance_table(&f.query);
        f.pq.adc_distances(&table, &f.codes, &mut out);
        black_box(out.as_mut_ptr());
    });
    Measurement { iterations, elapsed, bytes_per_vector: Some(f.m) }
}

/// Fixed per-query overhead of ADC: k * dim multiply-adds.
fn adc_table_only(f: &Fixture, min_time: Duration) -> Measurement {
    let (iterations, elapsed) = measure(min_time, || {
        black_box(f.pq.distance_table(&f.query));
    });
    Measurement { iterations, elapsed, bytes_per_vector: None }
}

/// SQ8 scan: decode each byte on the fly and accumulate squared differences.
fn sq_scan(f: &Fixture, min_time: Duration) -> Measurement {
    let mut out = vec![0.0; f.n];
    let (iterations, elapsed) = measure(min_time, || {
        f.sq.distances(&f.query, &f.sq_codes, &mut out);
        black_box(out.as_mut_ptr());
    });
    Measurement { iterations, elapsed, bytes_per_vector: Some(f.dim) }
}

// --- Single distance computations (latency, everything hot in L1) ---------
// Only the first vector / code of the fixture is used. black_box on the inputs
// stops the compiler from hoisting the computation out of the loop, since the
// inputs never change.

fn single_exact(f: &Fixture, min_time: Duration) -> Measurement {
    let (q, x) = (&f.query[..], &f.data[..f.dim]);
    let (iterations, elapsed) = measure(min_time, || {
        black_box(l2_distance(black_box(q), black_box(x)));
    });
    Measurement { iterations, elapsed, bytes_per_vector: None }
}

/// One ADC distance with the table already built: m lookups + adds.
fn single_adc(f: &Fixture, min_time: Duration) -> Measurement {
    let (table, code) = (&f.table[..], &f.codes[..f.m]);
    let (iterations, elapsed) = measure(min_time, || {
        let mut d = 0.0;
        f.pq.adc_distances(black_box(table), black_box(code), slice::from_mut(&mut d));
        black_box(d);
    });
    Measurement { iterations, elapsed, bytes_per_vector: None }
}

/// One ADC distance including building the table: what a single
/// query-to-single-vector comparison really costs with PQ.
fn single_adc_with_table(f: &Fixture, min_time: Duration) -> Measurement {
    let (q, code) = (&f.query[..], &f.codes[..f.m]);
    let (iterations, elapsed) = measure(min_time, || {
        let table = f.pq.distance_table(black_box(q));
        let mut d = 0.0;
        f.pq.adc_distances(&table, black_box(code), slice::from_mut(&mut d));
        black_box(d);
    });
    Measurement { iterations, elapsed, bytes_per_vector: None }
}

fn single_sq(f: &Fixture, min_time: Duration) -> Measurement {
    let (q, code) = (&f.query[..], &f.sq_codes[..f.dim]);
    let (iterations, elapsed) = measure(min_time, || {
        let mut d = 0.0;
        f.sq.distances(black_box(q), black_box(code), slice::from_mut(&mut d));
        black_box(d);
    });
    Measurement { iterations, elapsed, bytes_per_vector: None }
}

struct Options {
    filter: Option<String>,
    counters: bool,
    min_time: Duration,
}

fn parse_args() -> Result<Options, String> {
    let mut options =
        Options { filter: None, counters: false, min_time: Duration::from_secs_f64(0.5) };
    for arg in std::env::args().skip(1) {
        if arg == "--bench" {
            // cargo bench passes this to every target.
        } else if arg == "--counters" {
            options.counters = true;
        } else if let Some(seconds) = arg.strip_prefix("--min-time=") {
            let seconds = seconds.parse().map_err(|_| format!("bad value in {arg}"))?;
            options.min_time =
                Duration::try_from_secs_f64(seconds).map_err(|_| format!("bad value in {arg}"))?;
        } else if arg.starts_with('-') {
            return Err(format!("unrecognized argument: {arg}"));
        } else if options.filter.replace(arg).is_some() {
            return Err("at most one filter is allowed".to_owned());
        }
    }
    Ok(options)
}

/// Prints a value with a k/M/G suffix.
fn human(value: f64) -> String {
    match value {
        v if v >= 1e9 => format!("{:.3}G", v / 1e9),
        v if v >= 1e6 => format!("{:.3}M", v / 1e6),
        v if v >= 1e3 => format!("{:.3}k", v / 1e3),
        v => format!("{v:.3}"),
    }
}

/// Runs one benchmark and prints its row. `unit` is a label and its size in
/// seconds. Fixtures are built on first use, so filtered-out sizes cost nothing.
fn run(
    name: String,
    (dim, n): (usize, usize),
    unit: (&str, f64),
    benchmark: Benchmark,
    fixtures: &mut HashMap<(usize, usize), Fixture>,
    options: &Options,
) {
    if options.filter.as_ref().is_some_and(|filter| !name.contains(filter.as_str())) {
        return;
    }
    let fixture = fixtures.entry((dim, n)).or_insert_with(|| Fixture::new(dim, n));
    let result = benchmark(fixture, options.min_time);
    let seconds = result.elapsed.as_secs_f64();
    let per_iteration = seconds / result.iterations as f64;
    print!("{name:<40} {:>12.1} {:<2} {:>10}", per_iteration / unit.1, unit.0, result.iterations);
    if let (true, Some(bytes)) = (options.counters, result.bytes_per_vector) {
        let items = (result.iterations * n as u64) as f64 / seconds;
        print!(
            "  bytes/s={}  bytes/vec={bytes}  items/s={}",
            human(items * bytes as f64),
            human(items)
        );
    }
    println!();
}

fn main() {
    // `cargo test --all-targets` also runs this binary, unoptimized.
    if cfg!(debug_assertions) {
        println!("skipping benchmarks: they need optimizations (use cargo bench)");
        return;
    }
    let options = match parse_args() {
        Ok(options) => options,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(1);
        }
    };
    let mut fixtures = HashMap::new();

    println!("{:<40} {:>15} {:>10}", "Benchmark", "Time", "Iterations");
    let scans: [(&str, Benchmark); 5] = [
        ("BM_ExactL2", exact_l2),
        ("BM_ADC_Scan", adc_scan),
        ("BM_ADC_TablePlusScan", adc_table_plus_scan),
        ("BM_ADC_TableOnly", adc_table_only),
        ("BM_SQ_Scan", sq_scan),
    ];
    for (name, benchmark) in scans {
        for dim in DIMS {
            for n in SIZES {
                let name = format!("{name}/dim:{dim}/n:{n}");
                run(name, (dim, n), ("us", 1e-6), benchmark, &mut fixtures, &options);
            }
        }
    }
    let singles: [(&str, Benchmark); 4] = [
        ("BM_Single_Exact", single_exact),
        ("BM_Single_ADC", single_adc),
        ("BM_Single_ADCWithTable", single_adc_with_table),
        ("BM_Single_SQ", single_sq),
    ];
    for (name, benchmark) in singles {
        for dim in DIMS {
            let name = format!("{name}/dim:{dim}");
            run(name, (dim, SINGLE_N), ("ns", 1e-9), benchmark, &mut fixtures, &options);
        }
    }
}
