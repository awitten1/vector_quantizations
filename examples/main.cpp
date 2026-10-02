// Minimal product quantization walkthrough: train, encode, decode, search.
#include <cstdio>
#include <random>
#include <vector>

#include "vq/pq.hpp"

int main() {
  // 1. Data: n vectors of dimension dim, stored row-major in one flat array.
  //    Vector i lives at data[i * dim .. i * dim + dim).
  const int n = 1000, dim = 8;
  std::vector<float> data(n * dim);
  std::mt19937_64 rng(0);
  std::normal_distribution<float> normal;
  for (float& x : data) x = normal(rng);

  // 2. Train: split each vector into m = 4 pieces of dsub = 2 numbers,
  //    and learn k = 16 centroids per piece with k-means.
  const int m = 4, k = 16, iters = 20;
  vq::ProductQuantizer<float> pq(data.data(), n, dim, m, k, iters);
  std::printf("trained: dim=%d m=%d k=%d dsub=%d\n", pq.dim(), pq.m(), pq.k(), pq.dsub());

  // 3. Encode one vector: each piece becomes the index of its nearest centroid.
  const float* v = data.data();  // vector 0
  std::vector<uint8_t> code = pq.encode(v);
  std::printf("\nvector 0: ");
  for (int d = 0; d < dim; ++d) std::printf("%6.2f ", v[d]);
  std::printf("\ncode:     ");
  for (uint8_t c : code) std::printf("%u ", c);
  std::printf("  (%d bytes instead of %zu)\n", m, dim * sizeof(float));

  // 4. Decode: look the centroids back up to get an approximation.
  std::vector<float> approx(dim);
  pq.decode(code.data(), approx.data());
  std::printf("decoded:  ");
  for (int d = 0; d < dim; ++d) std::printf("%6.2f ", approx[d]);
  std::printf("\nsquared error: %.3f\n", vq::l2_distance(v, approx.data(), dim));

  // 5. Search: encode the whole dataset, then find the vectors closest to a query
  //    using only the codes (asymmetric distance via a lookup table).
  std::vector<uint8_t> codes = pq.encode(data.data(), n);
  std::vector<float> query(dim);
  for (float& x : query) x = normal(rng);

  std::printf("\ntop 5 for a random query (approx distance vs exact distance):\n");
  for (auto [id, approx_dist] : pq.search(query.data(), codes, 5)) {
    float exact = vq::l2_distance(query.data(), data.data() + id * dim, dim);
    std::printf("  id %4d  approx %.3f  exact %.3f\n", id, approx_dist, exact);
  }
  return 0;
}
