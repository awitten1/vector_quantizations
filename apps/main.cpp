#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "vq/vq.hpp"

int main() {
  constexpr int kN = 10000, kDim = 64, kM = 8, kK = 256, kIters = 25;
  constexpr int kQueries = 100, kTopN = 10;

  std::cout << "vector_quantizations v" << vq::version() << '\n';

  std::mt19937_64 rng(0);
  std::normal_distribution<float> dist;
  std::vector<float> data(static_cast<size_t>(kN) * kDim);
  for (auto& x : data) x = dist(rng);

  auto t0 = std::chrono::steady_clock::now();
  vq::ProductQuantizer<float> pq(data.data(), kN, kDim, kM, kK, kIters);
  auto t1 = std::chrono::steady_clock::now();

  std::vector<uint8_t> codes = pq.encode(data.data(), kN);
  double mse = 0;
  std::vector<float> decoded(kDim);
  for (int i = 0; i < kN; ++i) {
    pq.decode(codes.data() + static_cast<size_t>(i) * kM, decoded.data());
    mse += vq::l2_distance(data.data() + static_cast<size_t>(i) * kDim, decoded.data(), kDim);
  }
  mse /= kN;

  std::vector<float> queries(static_cast<size_t>(kQueries) * kDim);
  for (auto& x : queries) x = dist(rng);
  double recall = 0;
  for (int q = 0; q < kQueries; ++q) {
    const float* query = queries.data() + static_cast<size_t>(q) * kDim;
    std::vector<std::pair<float, int>> exact(kN);
    for (int i = 0; i < kN; ++i) exact[i] = {vq::l2_distance(query, data.data() + static_cast<size_t>(i) * kDim, kDim), i};
    std::partial_sort(exact.begin(), exact.begin() + kTopN, exact.end());
    auto approx = pq.search(query, codes, kTopN);
    for (auto [id, _] : approx) {
      for (int j = 0; j < kTopN; ++j) {
        if (exact[j].second == id) {
          recall += 1;
          break;
        }
      }
    }
  }
  recall /= static_cast<double>(kQueries) * kTopN;

  std::cout << "train: " << std::chrono::duration<double>(t1 - t0).count() << " s\n"
            << "bytes/vector: " << kM << " (" << (kDim * sizeof(float)) / kM << "x compression)\n"
            << "mean squared reconstruction error: " << mse << " (data variance/vector: " << kDim
            << ")\n"
            << "recall@" << kTopN << ": " << recall << '\n';
  return 0;
}
