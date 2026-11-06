# vector_quantizations

## Build

```sh
cargo build --release
cargo run --release                     # benchmark: compression, error, recall@10
cargo run --release --example pq_example  # small walkthrough of train / encode / decode / search
```

## Test

```sh
cargo test
```

## Benchmark

```sh
cargo bench                       # distance throughput: exact L2 vs ADC vs SQ8
cargo bench -- ADC_Scan           # only benchmarks whose name contains a substring
cargo bench -- --counters         # also report items/s, bytes/s and bytes per vector
cargo bench -- --min-time=0.01    # seconds per benchmark (default 0.5)
```
