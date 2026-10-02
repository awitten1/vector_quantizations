#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#include "vq/vq.hpp"

namespace {

constexpr int kN = 10000, kDim = 64, kQueries = 100, kTopN = 10;

using SearchFn = std::function<std::vector<std::pair<int, float>>(const float* query)>;
using DecodeFn = std::function<void(int i, float* out)>;

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// Prints compression, mean squared reconstruction error and recall@kTopN
// against the exact top-kTopN for each query.
void report(const char* name, double train_s, int bytes_per_vector, const std::vector<float>& data,
            const std::vector<std::vector<int>>& exact_top, const std::vector<float>& queries,
            const DecodeFn& decode, const SearchFn& search) {
  double mse = 0;
  std::vector<float> decoded(kDim);
  for (int i = 0; i < kN; ++i) {
    decode(i, decoded.data());
    mse += vq::l2_distance(data.data() + static_cast<size_t>(i) * kDim, decoded.data(), kDim);
  }
  mse /= kN;

  double recall = 0;
  for (int q = 0; q < kQueries; ++q) {
    const auto& truth = exact_top[q];
    for (auto [id, _] : search(queries.data() + static_cast<size_t>(q) * kDim)) {
      recall += std::find(truth.begin(), truth.end(), id) != truth.end();
    }
  }
  recall /= static_cast<double>(kQueries) * kTopN;

  std::cout << '\n' << name << '\n'
            << "  train: " << train_s << " s\n"
            << "  bytes/vector: " << bytes_per_vector << " ("
            << (kDim * sizeof(float)) / bytes_per_vector << "x compression)\n"
            << "  mean squared reconstruction error: " << mse << " (data variance/vector: " << kDim
            << ")\n"
            << "  recall@" << kTopN << ": " << recall << '\n';
}

}  // namespace

int main() {
  constexpr int kM = 8, kK = 256, kIters = 25;

  std::cout << "vector_quantizations v" << vq::version() << '\n';

  std::mt19937_64 rng(0);
  std::normal_distribution<float> dist;
  std::vector<float> data(static_cast<size_t>(kN) * kDim);
  for (auto& x : data) x = dist(rng);
  std::vector<float> queries(static_cast<size_t>(kQueries) * kDim);
  for (auto& x : queries) x = dist(rng);

  // Exact top-kTopN per query, shared by both quantizers.
  std::vector<std::vector<int>> exact_top(kQueries);
  for (int q = 0; q < kQueries; ++q) {
    const float* query = queries.data() + static_cast<size_t>(q) * kDim;
    std::vector<float> dists(kN);
    for (int i = 0; i < kN; ++i) {
      dists[i] = vq::l2_distance(query, data.data() + static_cast<size_t>(i) * kDim, kDim);
    }
    for (auto [id, _] : vq::top_n(dists, kTopN)) exact_top[q].push_back(id);
  }

  auto t0 = std::chrono::steady_clock::now();
  vq::ProductQuantizer<float> pq(data.data(), kN, kDim, kM, kK, kIters);
  double pq_train = seconds_since(t0);
  std::vector<uint8_t> pq_codes = pq.encode(data.data(), kN);
  report("product quantization (m=8, k=256)", pq_train, kM, data, exact_top, queries,
         [&](int i, float* out) { pq.decode(pq_codes.data() + static_cast<size_t>(i) * kM, out); },
         [&](const float* q) { return pq.search(q, pq_codes, kTopN); });

  t0 = std::chrono::steady_clock::now();
  vq::ScalarQuantizer<float> sq(data.data(), kN, kDim);
  double sq_train = seconds_since(t0);
  std::vector<uint8_t> sq_codes = sq.encode(data.data(), kN);
  report("scalar quantization (8-bit, per-dimension range)", sq_train, kDim, data, exact_top,
         queries,
         [&](int i, float* out) { sq.decode(sq_codes.data() + static_cast<size_t>(i) * kDim, out); },
         [&](const float* q) { return sq.search(q, sq_codes, kTopN); });
  return 0;
}
