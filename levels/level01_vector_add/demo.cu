#include "level01.cuh"

#include "dojo/cuda_utils.cuh"

#include <cstdio>
#include <numeric>
#include <vector>

// Level 1 demo: runs vector_add, saxpy, and reduce_sum on a sizable array,
// prints a sanity check, and reports effective memory bandwidth. Run it under
// `nsys profile` later to see the same numbers from the tool's perspective.
int main() {
  dojo::print_device_info();

  const int n = 1 << 24; // ~16.7M elements
  std::printf("\nN = %d elements (%.1f MB per array)\n", n,
              n * sizeof(float) / (1024.0 * 1024.0));

  std::vector<float> a(n), b(n), out(n);
  for (int i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i % 100);
    b[i] = 1.0f;
  }

  // ---- vector add ----
  {
    dojo::GpuTimer t;
    t.start();
    dojo::level01::vector_add(a.data(), b.data(), out.data(), n);
    float ms = t.stop();
    // Note: this time includes H2D + kernel + D2H since vector_add owns the
    // full flow. 3 array transfers + 1 store touched.
    std::size_t bytes = 4ull * n * sizeof(float);
    std::printf("\nvector_add: out[123] = %.1f (expect %.1f), %.3f ms, ~%.1f GB/s\n",
                out[123], a[123] + b[123], ms, dojo::gbps(bytes, ms));
  }

  // ---- saxpy ----
  {
    const float alpha = 2.5f;
    dojo::GpuTimer t;
    t.start();
    dojo::level01::saxpy(alpha, a.data(), b.data(), out.data(), n);
    float ms = t.stop();
    std::printf("saxpy:      out[123] = %.1f (expect %.1f), %.3f ms\n",
                out[123], alpha * a[123] + b[123], ms);
  }

  // ---- reduction ----
  {
    dojo::GpuTimer t;
    t.start();
    float gpu_sum = dojo::level01::reduce_sum(a.data(), n);
    float ms = t.stop();
    double cpu_sum = std::accumulate(a.begin(), a.end(), 0.0);
    std::printf("reduce_sum: gpu = %.0f, cpu = %.0f, %.3f ms\n", gpu_sum,
                cpu_sum, ms);
  }

  return 0;
}
