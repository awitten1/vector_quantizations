#pragma once

#include <algorithm>
#include <utility>
#include <vector>

namespace vq {

template<typename Float>
Float l2_distance(const Float* v1, const Float* v2, int dim) {
  Float distance = 0;
  for (int i = 0; i < dim; ++i) {
    auto diff = (v1[i] - v2[i]);
    distance += diff * diff;
  }
  return distance;
}

// Indices and values of the n smallest distances, ascending.
template<typename Float>
std::vector<std::pair<int, Float>> top_n(const std::vector<Float>& distances, int n) {
  const int count = static_cast<int>(distances.size());
  std::vector<std::pair<int, Float>> results(count);
  for (int i = 0; i < count; ++i) {
    results[i] = {i, distances[i]};
  }
  n = std::min(n, count);
  std::partial_sort(results.begin(), results.begin() + n, results.end(),
                    [](const auto& a, const auto& b) { return a.second < b.second; });
  results.resize(n);
  return results;
}

}  // namespace vq
