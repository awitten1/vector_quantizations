#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "vq/common.hpp"

namespace vq {

// k-means++ seeding over n contiguous dim-dimensional points in `data`.
// Writes k centroids to `centroids` (k * dim). Each new centroid is a training
// point sampled with probability proportional to its squared distance to the
// nearest centroid chosen so far. Requires n >= 1.
template<typename Float>
void kmeans_plus_plus(const Float* data, int n, int dim, int k, Float* centroids,
                      std::mt19937_64& rng) {
  std::uniform_int_distribution<int> pick(0, n - 1);
  std::vector<double> min_dist(n, std::numeric_limits<double>::max());
  int chosen = pick(rng);
  for (int c = 0; c < k; ++c) {
    const Float* src = data + static_cast<size_t>(chosen) * dim;
    Float* centroid = centroids + c * dim;
    std::copy(src, src + dim, centroid);
    if (c + 1 == k) {
      break;
    }
    for (int v = 0; v < n; ++v) {
      const Float* x = data + static_cast<size_t>(v) * dim;
      min_dist[v] = std::min<double>(min_dist[v], l2_distance(x, centroid, dim));
    }
    double total = std::accumulate(min_dist.begin(), min_dist.end(), 0.0);
    if (total > 0) {
      std::discrete_distribution<int> weighted(min_dist.begin(), min_dist.end());
      chosen = weighted(rng);
    } else {
      chosen = pick(rng);  // every point already coincides with a centroid
    }
  }
}

// Product quantizer: splits each dim-dimensional vector into m subvectors of
// dsub = dim / m and learns a k-centroid codebook per subspace with k-means.
// Each vector is then stored as m one-byte codes.
//
// All vector inputs are row-major: vector i of a batch is data[i * dim, (i + 1) * dim).
template<typename Float>
class ProductQuantizer {

  int dim_;
  int m_;
  int k_;
  int iters_;
  int dsub_;
  std::vector<Float> codebooks_;  // [m][k][dsub]

public:
  // n = number of vectors
  // dim = dimension of vectors
  // data = all n vectors stored contiguously.  dim * n floats.
  // m = number of subspaces (m should divide dim)
  // k = number of centroids
  ProductQuantizer(const Float* data, int n, int dim, int m, int k, int iters,
                   uint64_t seed = 42)
      : dim_(dim), m_(m), k_(k), iters_(iters) {
    if (dim <= 0 || m <= 0 || dim % m != 0) {
      throw std::invalid_argument("dim must be a positive multiple of m");
    }
    if (k <= 0 || k > 256) {
      throw std::invalid_argument("k must be in [1, 256]");
    }
    if (n < k) {
      throw std::invalid_argument("need at least k training vectors");
    }
    dsub_ = dim / m;
    codebooks_.resize(static_cast<size_t>(m_) * k_ * dsub_);

    // Gather each subspace into a contiguous n * dsub buffer so k-means
    // streams through memory instead of striding across full vectors.
    std::vector<Float> sub_data(static_cast<size_t>(n) * dsub_);
    for (int sub = 0; sub < m_; ++sub) {
      for (int v = 0; v < n; ++v) {
        const Float* src = data + static_cast<size_t>(v) * dim_ + sub * dsub_;
        std::copy(src, src + dsub_, sub_data.data() + static_cast<size_t>(v) * dsub_);
      }
      kmeans(sub_data.data(), n, sub, seed + sub);
    }
  }

  int dim() const { return dim_; }
  int m() const { return m_; }
  int k() const { return k_; }
  int dsub() const { return dsub_; }

  const Float* centroid(int sub, int c) const {
    return codebooks_.data() + (static_cast<size_t>(sub) * k_ + c) * dsub_;
  }

  std::vector<uint8_t> encode(const Float* vec) const {
    std::vector<uint8_t> codes(m_);
    for (int sub = 0; sub < m_; ++sub) {
      codes[sub] = static_cast<uint8_t>(nearest(vec + sub * dsub_, sub));
    }
    return codes;
  }

  // Encodes n contiguous vectors into a flat array of n * m codes.
  std::vector<uint8_t> encode(const Float* data, int n) const {
    std::vector<uint8_t> codes(static_cast<size_t>(n) * m_);
    for (int v = 0; v < n; ++v) {
      const Float* vec = data + static_cast<size_t>(v) * dim_;
      for (int sub = 0; sub < m_; ++sub) {
        codes[static_cast<size_t>(v) * m_ + sub] = static_cast<uint8_t>(nearest(vec + sub * dsub_, sub));
      }
    }
    return codes;
  }

  void decode(const uint8_t* codes, Float* out) const {
    for (int sub = 0; sub < m_; ++sub) {
      const Float* c = centroid(sub, codes[sub]);
      std::copy(c, c + dsub_, out + sub * dsub_);
    }
  }

  // ADC lookup table: table[sub * k + c] = ||query_sub - centroid(sub, c)||^2.
  std::vector<Float> distance_table(const Float* query) const {
    std::vector<Float> table(static_cast<size_t>(m_) * k_);
    for (int sub = 0; sub < m_; ++sub) {
      for (int c = 0; c < k_; ++c) {
        table[sub * k_ + c] = l2_distance(query + sub * dsub_, centroid(sub, c), dsub_);
      }
    }
    return table;
  }

  // Asymmetric distances from a query to num_codes encoded vectors, using the
  // query's distance_table: out[i] = sum over sub of table[sub * k + code_i[sub]].
  void adc_distances(const Float* table, const uint8_t* codes, int num_codes, Float* out) const {
    for (int i = 0; i < num_codes; ++i) {
      const uint8_t* code = codes + static_cast<size_t>(i) * m_;
      Float distance = 0;
      for (int sub = 0; sub < m_; ++sub) {
        distance += table[sub * k_ + code[sub]];
      }
      out[i] = distance;
    }
  }

  // Top-n nearest encoded vectors by asymmetric distance, ascending.
  // codes is a flat array of num_codes * m bytes.
  std::vector<std::pair<int, Float>> search(const Float* query, const std::vector<uint8_t>& codes,
                                            int n) const {
    const int num_codes = static_cast<int>(codes.size() / m_);
    const std::vector<Float> table = distance_table(query);
    std::vector<Float> distances(num_codes);
    adc_distances(table.data(), codes.data(), num_codes, distances.data());
    return top_n(distances, n);
  }

private:

  int nearest(const Float* subvec, int sub) const {
    int closest = 0;
    Float closest_distance = std::numeric_limits<Float>::max();
    for (int c = 0; c < k_; ++c) {
      Float distance = l2_distance(subvec, centroid(sub, c), dsub_);
      if (distance < closest_distance) {
        closest_distance = distance;
        closest = c;
      }
    }
    return closest;
  }

  // Lloyd's k-means for subspace `sub` on n contiguous dsub-dimensional points.
  void kmeans(const Float* data, int n, int sub, uint64_t seed) {
    Float* centroids = codebooks_.data() + static_cast<size_t>(sub) * k_ * dsub_;
    std::mt19937_64 rng(seed);

    kmeans_plus_plus(data, n, dsub_, k_, centroids, rng);

    std::uniform_int_distribution<int> pick(0, n - 1);  // for re-seeding empty clusters
    std::vector<int> assign(n, -1);
    std::vector<int> counts(k_);
    for (int i = 0; i < iters_; ++i) {
      // Assignment step.
      bool changed = false;
      for (int v = 0; v < n; ++v) {
        int closest = nearest(data + static_cast<size_t>(v) * dsub_, sub);
        if (closest != assign[v]) {
          assign[v] = closest;
          changed = true;
        }
      }
      if (!changed) {
        break;
      }

      // Update step: each centroid becomes the mean of its members.
      std::fill(centroids, centroids + k_ * dsub_, Float(0));
      std::fill(counts.begin(), counts.end(), 0);
      for (int v = 0; v < n; ++v) {
        Float* c = centroids + assign[v] * dsub_;
        const Float* x = data + static_cast<size_t>(v) * dsub_;
        for (int d = 0; d < dsub_; ++d) {
          c[d] += x[d];
        }
        ++counts[assign[v]];
      }
      for (int c = 0; c < k_; ++c) {
        Float* centroid = centroids + c * dsub_;
        if (counts[c] == 0) {
          // Empty cluster: re-seed from a random training point.
          const Float* src = data + static_cast<size_t>(pick(rng)) * dsub_;
          std::copy(src, src + dsub_, centroid);
          continue;
        }
        for (int d = 0; d < dsub_; ++d) {
          centroid[d] /= counts[c];
        }
      }
    }
  }

};

}  // namespace vq
