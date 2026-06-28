#include "level07.cuh"

#include "dojo/cuda_utils.cuh"

#include <cstdio>
#include <numeric>
#include <vector>

// Level 7 demo and profiling target: warp-shuffle reduction and warp-aggregated
// histogram.
//
// What to look for under the profilers:
//   ncu --set full ./level07_demo
//   - reduce_sum_warp is bandwidth-bound (it reads the input once). Compare its
//     GB/s and its __syncthreads() count against Level 1's reduce_sum — the
//     shuffle version should hit similar bandwidth with one barrier per block.
//   - For the histogram, once you implement aggregation, watch the L2/atomic
//     throughput on a low-entropy input (few hot bins): naive serializes the
//     same-bin atomics; aggregated issues one per warp-group.
int main() {
  dojo::print_device_info();

  // ---- warp reduction (worked example) ----
  {
    const int n = 1 << 24; // ~16.7M floats
    std::vector<float> a(n);
    for (int i = 0; i < n; ++i) {
      a[i] = static_cast<float>(i % 100);
    }

    dojo::GpuTimer t;
    t.start();
    float gpu = dojo::level07::reduce_sum_warp(a.data(), n);
    float ms = t.stop();
    double cpu = std::accumulate(a.begin(), a.end(), 0.0);
    std::printf("\nreduce_sum_warp: gpu = %.0f, cpu = %.0f, %.3f ms, ~%.1f GB/s\n",
                gpu, cpu, ms, dojo::gbps(static_cast<std::size_t>(n) * sizeof(float), ms));
  }

  // ---- warp histogram (stub: bins stay zero until implemented) ----
  {
    const int n = 1 << 24;
    const int num_bins = 256;
    std::vector<unsigned char> in(n);
    for (int i = 0; i < n; ++i) {
      in[i] = static_cast<unsigned char>((i * 31) % 256);
    }
    std::vector<unsigned int> bins(num_bins, 0);

    dojo::GpuTimer t;
    t.start();
    dojo::level07::histogram_warp(in.data(), n, bins.data(), num_bins);
    float ms = t.stop();

    unsigned long long total = 0;
    for (unsigned int c : bins) {
      total += c;
    }
    std::printf("histogram_warp:  counted %llu of %d elements, %.3f ms (stub: expect 0)\n",
                total, n, ms);
  }

  return 0;
}
