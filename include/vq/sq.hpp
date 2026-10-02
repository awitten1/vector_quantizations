#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "vq/common.hpp"

namespace vq {

// 8-bit scalar quantizer with a separate range per dimension, as in FAISS
// ScalarQuantizer QT_8bit with RS_minmax. Training records each dimension's
// [min, max] and splits it into 256 equal buckets; a value is stored as its
// bucket index and reconstructed at the bucket center.
//
// All vector inputs are row-major: vector i of a batch is data[i * dim, (i + 1) * dim).
template<typename Float>
class ScalarQuantizer {

  static constexpr int kLevels = 256;

  int dim_;
  std::vector<Float> min_;   // per dimension
  std::vector<Float> step_;  // per dimension: (max - min) / 256

public:
  // Trains on n vectors stored contiguously in `data` (n * dim values).
  ScalarQuantizer(const Float* data, int n, int dim) : dim_(dim) {
    if (dim <= 0) {
      throw std::invalid_argument("dim must be positive");
    }
    if (n <= 0) {
      throw std::invalid_argument("need at least one training vector");
    }
    min_.assign(data, data + dim_);
    std::vector<Float> max(data, data + dim_);
    for (int v = 1; v < n; ++v) {
      const Float* x = data + static_cast<size_t>(v) * dim_;
      for (int d = 0; d < dim_; ++d) {
        min_[d] = std::min(min_[d], x[d]);
        max[d] = std::max(max[d], x[d]);
      }
    }
    step_.resize(dim_);
    for (int d = 0; d < dim_; ++d) {
      step_[d] = (max[d] - min_[d]) / kLevels;
    }
  }

  int dim() const { return dim_; }

  std::vector<uint8_t> encode(const Float* vec) const {
    std::vector<uint8_t> code(dim_);
    encode_into(vec, code.data());
    return code;
  }

  // Encodes n contiguous vectors into a flat array of n * dim codes.
  std::vector<uint8_t> encode(const Float* data, int n) const {
    std::vector<uint8_t> codes(static_cast<size_t>(n) * dim_);
    for (int v = 0; v < n; ++v) {
      encode_into(data + static_cast<size_t>(v) * dim_, codes.data() + static_cast<size_t>(v) * dim_);
    }
    return codes;
  }

  void decode(const uint8_t* code, Float* out) const {
    for (int d = 0; d < dim_; ++d) {
      out[d] = reconstruct(code[d], d);
    }
  }

  // Asymmetric distances: out[i] = ||query - decode(code_i)||^2, decoding on the fly.
  void distances(const Float* query, const uint8_t* codes, int num_codes, Float* out) const {
    for (int i = 0; i < num_codes; ++i) {
      const uint8_t* code = codes + static_cast<size_t>(i) * dim_;
      Float distance = 0;
      for (int d = 0; d < dim_; ++d) {
        Float diff = query[d] - reconstruct(code[d], d);
        distance += diff * diff;
      }
      out[i] = distance;
    }
  }

  // Top-n nearest encoded vectors by asymmetric distance, ascending.
  // codes is a flat array of num_codes * dim bytes.
  std::vector<std::pair<int, Float>> search(const Float* query, const std::vector<uint8_t>& codes,
                                            int n) const {
    const int num_codes = static_cast<int>(codes.size() / dim_);
    std::vector<Float> dists(num_codes);
    distances(query, codes.data(), num_codes, dists.data());
    return top_n(dists, n);
  }

private:

  void encode_into(const Float* vec, uint8_t* code) const {
    for (int d = 0; d < dim_; ++d) {
      if (step_[d] == 0) {
        code[d] = 0;  // constant dimension
        continue;
      }
      Float bucket = std::floor((vec[d] - min_[d]) / step_[d]);
      code[d] = static_cast<uint8_t>(std::clamp<Float>(bucket, 0, kLevels - 1));
    }
  }

  Float reconstruct(uint8_t c, int d) const {
    return min_[d] + (c + Float(0.5)) * step_[d];
  }

};

}  // namespace vq
