#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include "check.hpp"
#include "vq/sq.hpp"

using vq::ScalarQuantizer;

namespace {

// n row-major dim-dimensional vectors.
struct Dataset {
  std::vector<float> data;
  int n;
  int dim;

  float* row(int i) { return data.data() + static_cast<size_t>(i) * dim; }
};

Dataset make_dataset(int n, int dim, uint64_t seed) {
  Dataset ds{std::vector<float>(static_cast<size_t>(n) * dim), n, dim};
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist;
  for (auto& x : ds.data) x = dist(rng);
  return ds;
}

// Per-dimension bucket width, computed independently of the quantizer.
std::vector<float> steps(Dataset& ds) {
  std::vector<float> step(ds.dim);
  for (int d = 0; d < ds.dim; ++d) {
    float lo = ds.row(0)[d], hi = lo;
    for (int i = 1; i < ds.n; ++i) {
      lo = std::min(lo, ds.row(i)[d]);
      hi = std::max(hi, ds.row(i)[d]);
    }
    step[d] = (hi - lo) / 256;
  }
  return step;
}

void test_rejects_bad_params() {
  Dataset ds = make_dataset(4, 4, 1);
  CHECK_THROWS(ScalarQuantizer<float>(ds.data.data(), 0, 4), std::invalid_argument);
  CHECK_THROWS(ScalarQuantizer<float>(ds.data.data(), 4, 0), std::invalid_argument);
}

void test_error_within_half_bucket() {
  Dataset ds = make_dataset(500, 16, 2);
  ScalarQuantizer<float> sq(ds.data.data(), ds.n, ds.dim);
  std::vector<float> step = steps(ds);
  std::vector<float> out(ds.dim);
  for (int i = 0; i < ds.n; ++i) {
    sq.decode(sq.encode(ds.row(i)).data(), out.data());
    for (int d = 0; d < ds.dim; ++d) {
      CHECK(std::abs(ds.row(i)[d] - out[d]) <= step[d] / 2 + 1e-5f);
    }
  }
}

void test_out_of_range_clamps() {
  Dataset ds = make_dataset(100, 4, 3);
  ScalarQuantizer<float> sq(ds.data.data(), ds.n, ds.dim);
  std::vector<float> low(4, -1e6f), high(4, 1e6f);
  for (uint8_t c : sq.encode(low.data())) CHECK(c == 0);
  for (uint8_t c : sq.encode(high.data())) CHECK(c == 255);
}

void test_constant_dimension_is_exact() {
  Dataset ds = make_dataset(50, 3, 4);
  for (int i = 0; i < ds.n; ++i) ds.row(i)[1] = 7.25f;
  ScalarQuantizer<float> sq(ds.data.data(), ds.n, ds.dim);
  std::vector<float> out(ds.dim);
  sq.decode(sq.encode(ds.row(0)).data(), out.data());
  CHECK(out[1] == 7.25f);
}

void test_distances_match_decoded() {
  Dataset ds = make_dataset(300, 16, 5);
  Dataset queries = make_dataset(5, 16, 6);
  ScalarQuantizer<float> sq(ds.data.data(), ds.n, ds.dim);
  std::vector<uint8_t> codes = sq.encode(ds.data.data(), ds.n);
  std::vector<float> dists(ds.n), decoded(ds.dim);
  for (int qi = 0; qi < queries.n; ++qi) {
    float* q = queries.row(qi);
    sq.distances(q, codes.data(), ds.n, dists.data());
    for (int i = 0; i < ds.n; ++i) {
      sq.decode(codes.data() + static_cast<size_t>(i) * ds.dim, decoded.data());
      CHECK_NEAR(dists[i], vq::l2_distance(q, decoded.data(), ds.dim), 1e-3f);
    }
  }
}

void test_search() {
  Dataset ds = make_dataset(200, 16, 7);
  ScalarQuantizer<float> sq(ds.data.data(), ds.n, ds.dim);
  std::vector<uint8_t> codes = sq.encode(ds.data.data(), ds.n);
  for (int i = 0; i < ds.n; ++i) {
    auto results = sq.search(ds.row(i), codes, 10);
    CHECK(results.size() == 10);
    CHECK(results[0].first == i);
    for (size_t j = 1; j < results.size(); ++j) CHECK(results[j - 1].second <= results[j].second);
  }
}

void test_batch_encode_matches_single() {
  Dataset ds = make_dataset(100, 8, 8);
  ScalarQuantizer<float> sq(ds.data.data(), ds.n, ds.dim);
  std::vector<uint8_t> batch = sq.encode(ds.data.data(), ds.n);
  CHECK(batch.size() == static_cast<size_t>(ds.n) * ds.dim);
  for (int i = 0; i < ds.n; ++i) {
    auto single = sq.encode(ds.row(i));
    CHECK(std::equal(single.begin(), single.end(), batch.begin() + static_cast<size_t>(i) * ds.dim));
  }
}

void test_double() {
  std::vector<double> data(64 * 4);
  std::mt19937_64 rng(9);
  std::normal_distribution<double> dist;
  for (auto& x : data) x = dist(rng);
  ScalarQuantizer<double> sq(data.data(), 64, 4);
  std::vector<double> out(4);
  sq.decode(sq.encode(data.data()).data(), out.data());
  CHECK(vq::l2_distance(data.data(), out.data(), 4) < 0.01);
}

}  // namespace

int main() {
  test_rejects_bad_params();
  test_error_within_half_bucket();
  test_out_of_range_clamps();
  test_constant_dimension_is_exact();
  test_distances_match_decoded();
  test_search();
  test_batch_encode_matches_single();
  test_double();
  std::cout << "all tests passed\n";
  return EXIT_SUCCESS;
}
