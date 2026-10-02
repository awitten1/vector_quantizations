#include <algorithm>
#include <random>
#include <stdexcept>
#include <vector>

#include "check.hpp"
#include "vq/pq.hpp"

using vq::ProductQuantizer;

namespace {

// n row-major dim-dimensional vectors.
struct Dataset {
  std::vector<float> data;
  int n;
  int dim;

  float* row(int i) { return data.data() + static_cast<size_t>(i) * dim; }
};

Dataset make_dataset(int n, int dim, uint64_t seed, float stddev = 1.0f) {
  Dataset ds{std::vector<float>(static_cast<size_t>(n) * dim), n, dim};
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0.0f, stddev);
  for (auto& x : ds.data) x = dist(rng);
  return ds;
}

// n points tightly scattered around `clusters` random centers.
Dataset make_clustered(int n, int dim, int clusters, uint64_t seed) {
  Dataset centers = make_dataset(clusters, dim, seed, 10.0f);
  Dataset ds = make_dataset(n, dim, seed + 1, 0.01f);
  for (int i = 0; i < n; ++i) {
    for (int d = 0; d < dim; ++d) ds.row(i)[d] += centers.row(i % clusters)[d];
  }
  return ds;
}

void test_kmeans_plus_plus_covers_clusters() {
  const int dim = 4, clusters = 8;
  Dataset ds = make_clustered(400, dim, clusters, 9);
  for (uint64_t seed = 0; seed < 10; ++seed) {
    std::mt19937_64 rng(seed);
    std::vector<float> centroids(clusters * dim);
    vq::kmeans_plus_plus(ds.data.data(), ds.n, dim, clusters, centroids.data(), rng);
    // Points i and i % clusters share a cluster, so the first `clusters` points
    // are one representative per cluster. Each must have a centroid nearby.
    std::vector<bool> covered(clusters, false);
    for (int c = 0; c < clusters; ++c) {
      for (int j = 0; j < clusters; ++j) {
        if (vq::l2_distance(centroids.data() + c * dim, ds.row(j), dim) < 0.01f) {
          covered[j] = true;
        }
      }
    }
    for (int j = 0; j < clusters; ++j) CHECK(covered[j]);
  }
}

void test_rejects_bad_params() {
  Dataset ds = make_dataset(16, 8, 1);
  CHECK_THROWS(ProductQuantizer<float>(ds.data.data(), ds.n, 8, 3, 4, 10), std::invalid_argument);
  CHECK_THROWS(ProductQuantizer<float>(ds.data.data(), ds.n, 8, 2, 257, 10), std::invalid_argument);
  CHECK_THROWS(ProductQuantizer<float>(ds.data.data(), ds.n, 8, 2, 32, 10), std::invalid_argument);
}

void test_does_not_modify_input() {
  Dataset ds = make_dataset(200, 8, 2);
  const std::vector<float> before = ds.data;
  ProductQuantizer<float> pq(ds.data.data(), ds.n, 8, 2, 16, 10);
  CHECK(ds.data == before);
}

void test_exact_when_k_equals_n() {
  Dataset ds = make_dataset(16, 8, 3);
  ProductQuantizer<float> pq(ds.data.data(), ds.n, 8, 4, 16, 10);
  std::vector<float> out(8);
  for (int i = 0; i < ds.n; ++i) {
    float* v = ds.row(i);
    pq.decode(pq.encode(v).data(), out.data());
    CHECK_NEAR(vq::l2_distance(v, out.data(), 8), 0.0f, 1e-10f);
  }
}

void test_clustered_reconstruction() {
  const int dim = 16;
  Dataset ds = make_clustered(500, dim, 8, 4);
  ProductQuantizer<float> pq(ds.data.data(), ds.n, dim, 4, 8, 25);
  std::vector<float> out(dim);
  for (int i = 0; i < ds.n; ++i) {
    float* v = ds.row(i);
    pq.decode(pq.encode(v).data(), out.data());
    // Noise is 0.01 per coord, so squared error should be ~dim * 1e-4.
    CHECK(vq::l2_distance(v, out.data(), dim) < 0.05f);
  }
}

void test_adc_matches_decoded_distance() {
  const int dim = 16;
  Dataset ds = make_dataset(300, dim, 5);
  Dataset queries = make_dataset(5, dim, 6);
  ProductQuantizer<float> pq(ds.data.data(), ds.n, dim, 4, 16, 15);
  std::vector<uint8_t> codes = pq.encode(ds.data.data(), ds.n);
  std::vector<float> decoded(dim);
  for (int qi = 0; qi < queries.n; ++qi) {
    float* q = queries.row(qi);
    auto results = pq.search(q, codes, 10);
    CHECK(results.size() == 10);
    for (size_t i = 0; i < results.size(); ++i) {
      if (i > 0) CHECK(results[i - 1].second <= results[i].second);
      auto [id, dist] = results[i];
      pq.decode(codes.data() + static_cast<size_t>(id) * pq.m(), decoded.data());
      CHECK_NEAR(dist, vq::l2_distance(q, decoded.data(), dim), 1e-3f);
    }
  }
}

void test_batch_encode_matches_single() {
  Dataset ds = make_dataset(100, 8, 10);
  ProductQuantizer<float> pq(ds.data.data(), ds.n, 8, 4, 16, 10);
  std::vector<uint8_t> batch = pq.encode(ds.data.data(), ds.n);
  CHECK(batch.size() == static_cast<size_t>(ds.n) * pq.m());
  for (int i = 0; i < ds.n; ++i) {
    auto single = pq.encode(ds.row(i));
    CHECK(std::equal(single.begin(), single.end(), batch.begin() + static_cast<size_t>(i) * pq.m()));
  }
}

void test_self_query_is_top1() {
  const int dim = 16;
  Dataset ds = make_clustered(64, dim, 64, 7);  // every point its own cluster
  ProductQuantizer<float> pq(ds.data.data(), ds.n, dim, 4, 64, 25);
  std::vector<uint8_t> codes = pq.encode(ds.data.data(), ds.n);
  for (int i = 0; i < 64; ++i) {
    CHECK(pq.search(ds.row(i), codes, 1)[0].first == i);
  }
}

void test_double() {
  std::vector<double> data(64 * 4);
  std::mt19937_64 rng(8);
  std::normal_distribution<double> dist;
  for (auto& x : data) x = dist(rng);
  ProductQuantizer<double> pq(data.data(), 64, 4, 2, 8, 10);
  CHECK(pq.encode(data.data()).size() == 2);
}

}  // namespace

int main() {
  test_kmeans_plus_plus_covers_clusters();
  test_rejects_bad_params();
  test_does_not_modify_input();
  test_exact_when_k_equals_n();
  test_clustered_reconstruction();
  test_adc_matches_decoded_distance();
  test_batch_encode_matches_single();
  test_self_query_is_top1();
  test_double();
  std::cout << "all tests passed\n";
  return EXIT_SUCCESS;
}
