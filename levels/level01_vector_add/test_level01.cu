#include "level01.cuh"

#include "dojo/test.hpp"

#include <numeric>
#include <vector>

using namespace dojo::level01;

DOJO_TEST_CASE(vector_add_matches_host) {
  const int n = 100000;
  std::vector<float> a(n), b(n), out(n, -1.0f);
  for (int i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(2 * i);
  }

  vector_add(a.data(), b.data(), out.data(), n);

  for (int i = 0; i < n; ++i) {
    DOJO_CHECK_CLOSE(out[i], a[i] + b[i], 1e-3);
  }
}

DOJO_TEST_CASE(saxpy_matches_host) {
  const int n = 50000;
  const float alpha = 3.0f;
  std::vector<float> x(n), y(n), out(n, -1.0f);
  for (int i = 0; i < n; ++i) {
    x[i] = static_cast<float>(i % 7);
    y[i] = 1.0f;
  }

  saxpy(alpha, x.data(), y.data(), out.data(), n);

  for (int i = 0; i < n; ++i) {
    DOJO_CHECK_CLOSE(out[i], alpha * x[i] + y[i], 1e-3);
  }
}

DOJO_TEST_CASE(reduce_sum_matches_host) {
  const int n = 1 << 20;
  std::vector<float> in(n, 1.0f); // sum should be exactly n
  float gpu = reduce_sum(in.data(), n);
  DOJO_CHECK_CLOSE(gpu, static_cast<float>(n), 1.0);
}

DOJO_TEST_CASE(reduce_sum_handles_non_power_of_two) {
  const int n = 1234567;
  std::vector<float> in(n);
  for (int i = 0; i < n; ++i) {
    in[i] = static_cast<float>(i % 3); // 0,1,2,0,1,2,...
  }
  double expected = std::accumulate(in.begin(), in.end(), 0.0);
  float gpu = reduce_sum(in.data(), n);
  // ~1.6M values summed in float: allow a small relative tolerance.
  DOJO_CHECK_CLOSE(gpu, expected, expected * 1e-4);
}

DOJO_TEST_MAIN()
