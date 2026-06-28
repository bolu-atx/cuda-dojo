#include "level07.cuh"

#include "dojo/test.hpp"

#include <numeric>
#include <vector>

using namespace dojo::level07;

// ---- reduce_sum_warp (complete worked example: should PASS) ----------------

DOJO_TEST_CASE(reduce_warp_power_of_two) {
  const int n = 1 << 20;
  std::vector<float> in(n, 1.0f); // sum should be exactly n
  float gpu = reduce_sum_warp(in.data(), n);
  DOJO_CHECK_CLOSE(gpu, static_cast<float>(n), 1.0);
}

DOJO_TEST_CASE(reduce_warp_non_power_of_two) {
  // n not a multiple of 32 -> the tail warp has dead lanes. They must contribute
  // the identity (0), not garbage from an out-of-range read.
  const int n = 1234567;
  std::vector<float> in(n);
  for (int i = 0; i < n; ++i) {
    in[i] = static_cast<float>(i % 3);
  }
  double expected = std::accumulate(in.begin(), in.end(), 0.0);
  float gpu = reduce_sum_warp(in.data(), n);
  DOJO_CHECK_CLOSE(gpu, expected, expected * 1e-4);
}

DOJO_TEST_CASE(reduce_warp_single_warp) {
  // Exactly one warp's worth of data: isolates warp_reduce_sum itself.
  const int n = 32;
  std::vector<float> in(n);
  for (int i = 0; i < n; ++i) {
    in[i] = static_cast<float>(i);
  }
  float gpu = reduce_sum_warp(in.data(), n); // 0+1+...+31 = 496
  DOJO_CHECK_CLOSE(gpu, 496.0f, 1e-3);
}

DOJO_TEST_CASE(reduce_warp_tail_block) {
  // Just over a warp (33): a partly-filled warp plus a stray lane.
  const int n = 33;
  std::vector<float> in(n, 2.0f);
  float gpu = reduce_sum_warp(in.data(), n);
  DOJO_CHECK_CLOSE(gpu, 66.0f, 1e-3);
}

DOJO_TEST_CASE(reduce_warp_single_element) {
  const int n = 1;
  std::vector<float> in = {7.5f};
  float gpu = reduce_sum_warp(in.data(), n);
  DOJO_CHECK_CLOSE(gpu, 7.5f, 1e-3);
}

// ---- histogram_warp (stub: fails until implemented) ------------------------

namespace {

std::vector<unsigned int> cpu_histogram(const std::vector<unsigned char> &in,
                                        int num_bins) {
  std::vector<unsigned int> bins(num_bins, 0);
  for (unsigned char v : in) {
    int b = static_cast<int>(v) * num_bins / 256;
    bins[b]++;
  }
  return bins;
}

} // namespace

DOJO_TEST_CASE(histogram_matches_host) {
  const int n = 100000;
  const int num_bins = 256;
  std::vector<unsigned char> in(n);
  for (int i = 0; i < n; ++i) {
    in[i] = static_cast<unsigned char>((i * 31) % 256);
  }
  std::vector<unsigned int> bins(num_bins, 0);

  histogram_warp(in.data(), n, bins.data(), num_bins);

  auto ref = cpu_histogram(in, num_bins);
  for (int b = 0; b < num_bins; ++b) {
    DOJO_CHECK(bins[b] == ref[b]);
  }
}

DOJO_TEST_CASE(histogram_hot_bin) {
  // Low-entropy input: every byte maps to the same bin. This is the worst case
  // for naive atomics and where warp aggregation wins most — and it must still
  // count correctly.
  const int n = 50000;
  const int num_bins = 16;
  std::vector<unsigned char> in(n, 200); // all land in one bin
  std::vector<unsigned int> bins(num_bins, 0);

  histogram_warp(in.data(), n, bins.data(), num_bins);

  auto ref = cpu_histogram(in, num_bins);
  for (int b = 0; b < num_bins; ++b) {
    DOJO_CHECK(bins[b] == ref[b]);
  }
}

DOJO_TEST_MAIN()
