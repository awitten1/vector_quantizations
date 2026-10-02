// Distance computation throughput: exact L2 over raw float vectors vs ADC
// (asymmetric distance computation) over PQ codes vs scalar-quantized (SQ8) codes.
//
// Each benchmark computes the distance from one query to every database vector
// and writes it to an output array, so both sides do the same job:
//   Exact: dim multiply-adds per vector, reading dim * 4 bytes per vector.
//   ADC:   m table lookups + adds per vector, reading m bytes per vector.
//   SQ:    dim decodes + multiply-adds per vector, reading dim bytes per vector.
// Args: {dim, n}. m = dim / 8 (dsub = 8), k = 256, so PQ codes are dim / 8 bytes.
// Pass --counters to also report items/s, bytes/s and bytes per vector.
#include <benchmark/benchmark.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include "vq/pq.hpp"
#include "vq/sq.hpp"

namespace {

constexpr int kDsub = 8;
constexpr int kK = 256;
constexpr int kTrainN = 2000;  // codebook quality doesn't affect speed
constexpr int kTrainIters = 3;

bool g_counters = false;  // set by --counters

struct Fixture {
  int dim, n, m;
  std::vector<float> data;    // n * dim raw vectors
  std::vector<float> query;   // dim
  std::vector<uint8_t> codes; // n * m
  vq::ProductQuantizer<float> pq;
  std::vector<float> table;   // m * k, for the query
  vq::ScalarQuantizer<float> sq;
  std::vector<uint8_t> sq_codes; // n * dim

  static std::vector<float> random_floats(size_t count, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal;
    std::vector<float> v(count);
    for (float& x : v) x = normal(rng);
    return v;
  }

  Fixture(int dim, int n)
      : dim(dim), n(n), m(dim / kDsub),
        data(random_floats(static_cast<size_t>(n) * dim, 1)),
        query(random_floats(dim, 2)),
        pq(data.data(), kTrainN, dim, m, kK, kTrainIters),
        sq(data.data(), kTrainN, dim) {
    // Random codes instead of encoding: scan cost depends only on the code
    // layout, and encoding n vectors would dominate setup time.
    std::mt19937_64 rng(3);
    std::uniform_int_distribution<int> byte(0, kK - 1);
    codes.resize(static_cast<size_t>(n) * m);
    for (uint8_t& c : codes) c = static_cast<uint8_t>(byte(rng));
    sq_codes.resize(static_cast<size_t>(n) * dim);
    for (uint8_t& c : sq_codes) c = static_cast<uint8_t>(byte(rng));
    table = pq.distance_table(query.data());
  }
};

// Builds (and caches) one fixture per {dim, n} so setup isn't repeated.
Fixture& get_fixture(int dim, int n) {
  static std::vector<std::unique_ptr<Fixture>> cache;
  for (auto& f : cache) {
    if (f->dim == dim && f->n == n) return *f;
  }
  cache.push_back(std::make_unique<Fixture>(dim, n));
  return *cache.back();
}

void set_counters(benchmark::State& state, const Fixture& f, size_t bytes_per_vector) {
  if (!g_counters) return;
  state.SetItemsProcessed(state.iterations() * f.n);
  state.SetBytesProcessed(state.iterations() * f.n * bytes_per_vector);
  state.counters["bytes/vec"] = static_cast<double>(bytes_per_vector);
}

// Baseline: exact squared L2 from the query to every raw vector.
void BM_ExactL2(benchmark::State& state) {
  Fixture& f = get_fixture(state.range(0), state.range(1));
  std::vector<float> out(f.n);
  for (auto _ : state) {
    for (int i = 0; i < f.n; ++i) {
      out[i] = vq::l2_distance(f.query.data(), f.data.data() + static_cast<size_t>(i) * f.dim, f.dim);
    }
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }
  set_counters(state, f, f.dim * sizeof(float));
}

// ADC scan only: lookup table already built for this query.
void BM_ADC_Scan(benchmark::State& state) {
  Fixture& f = get_fixture(state.range(0), state.range(1));
  std::vector<float> out(f.n);
  for (auto _ : state) {
    f.pq.adc_distances(f.table.data(), f.codes.data(), f.n, out.data());
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }
  set_counters(state, f, f.m);
}

// ADC end to end per query: build the m * k table, then scan.
void BM_ADC_TablePlusScan(benchmark::State& state) {
  Fixture& f = get_fixture(state.range(0), state.range(1));
  std::vector<float> out(f.n);
  for (auto _ : state) {
    std::vector<float> table = f.pq.distance_table(f.query.data());
    f.pq.adc_distances(table.data(), f.codes.data(), f.n, out.data());
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }
  set_counters(state, f, f.m);
}

// Fixed per-query overhead of ADC: k * dim multiply-adds.
void BM_ADC_TableOnly(benchmark::State& state) {
  Fixture& f = get_fixture(state.range(0), state.range(1));
  for (auto _ : state) {
    std::vector<float> table = f.pq.distance_table(f.query.data());
    benchmark::DoNotOptimize(table.data());
  }
}

// SQ8 scan: decode each byte on the fly and accumulate squared differences.
void BM_SQ_Scan(benchmark::State& state) {
  Fixture& f = get_fixture(state.range(0), state.range(1));
  std::vector<float> out(f.n);
  for (auto _ : state) {
    f.sq.distances(f.query.data(), f.sq_codes.data(), f.n, out.data());
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }
  set_counters(state, f, f.dim);
}

// --- Single distance computations (latency, everything hot in L1) ---------
// Reuses the n = 10k fixture; only the first vector / code is used.
// DoNotOptimize on the inputs stops the compiler from hoisting the
// computation out of the loop, since the inputs never change.

void BM_Single_Exact(benchmark::State& state) {
  Fixture& f = get_fixture(state.range(0), 10'000);
  const float* q = f.query.data();
  const float* x = f.data.data();
  for (auto _ : state) {
    benchmark::DoNotOptimize(q);
    benchmark::DoNotOptimize(x);
    float d = vq::l2_distance(q, x, f.dim);
    benchmark::DoNotOptimize(d);
  }
}

// One ADC distance with the table already built: m lookups + adds.
void BM_Single_ADC(benchmark::State& state) {
  Fixture& f = get_fixture(state.range(0), 10'000);
  const float* table = f.table.data();
  const uint8_t* code = f.codes.data();
  for (auto _ : state) {
    benchmark::DoNotOptimize(table);
    benchmark::DoNotOptimize(code);
    float d;
    f.pq.adc_distances(table, code, 1, &d);
    benchmark::DoNotOptimize(d);
  }
}

// One ADC distance including building the table: what a single
// query-to-single-vector comparison really costs with PQ.
void BM_Single_ADCWithTable(benchmark::State& state) {
  Fixture& f = get_fixture(state.range(0), 10'000);
  const float* q = f.query.data();
  const uint8_t* code = f.codes.data();
  for (auto _ : state) {
    benchmark::DoNotOptimize(q);
    benchmark::DoNotOptimize(code);
    std::vector<float> table = f.pq.distance_table(q);
    float d;
    f.pq.adc_distances(table.data(), code, 1, &d);
    benchmark::DoNotOptimize(d);
  }
}

void BM_Single_SQ(benchmark::State& state) {
  Fixture& f = get_fixture(state.range(0), 10'000);
  const float* q = f.query.data();
  const uint8_t* code = f.sq_codes.data();
  for (auto _ : state) {
    benchmark::DoNotOptimize(q);
    benchmark::DoNotOptimize(code);
    float d;
    f.sq.distances(q, code, 1, &d);
    benchmark::DoNotOptimize(d);
  }
}

void single_args(benchmark::Benchmark* b) {
  for (int dim : {64, 128, 256}) {
    b->Args({dim});
  }
  b->ArgNames({"dim"})->Unit(benchmark::kNanosecond);
}

void args(benchmark::Benchmark* b) {
  for (int dim : {64, 128, 256}) {
    for (int n : {10'000, 1'000'000}) {
      b->Args({dim, n});
    }
  }
  b->ArgNames({"dim", "n"})->Unit(benchmark::kMicrosecond);
}

}  // namespace

BENCHMARK(BM_ExactL2)->Apply(args);
BENCHMARK(BM_ADC_Scan)->Apply(args);
BENCHMARK(BM_ADC_TablePlusScan)->Apply(args);
BENCHMARK(BM_ADC_TableOnly)->Apply(args);
BENCHMARK(BM_SQ_Scan)->Apply(args);
BENCHMARK(BM_Single_Exact)->Apply(single_args);
BENCHMARK(BM_Single_ADC)->Apply(single_args);
BENCHMARK(BM_Single_ADCWithTable)->Apply(single_args);
BENCHMARK(BM_Single_SQ)->Apply(single_args);

int main(int argc, char** argv) {
  // Strip our own flag before Google Benchmark parses the rest.
  int out = 1;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--counters") == 0) {
      g_counters = true;
    } else {
      argv[out++] = argv[i];
    }
  }
  argc = out;
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
