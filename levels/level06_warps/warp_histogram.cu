#include "level06.cuh"

#include "dojo/cuda_utils.cuh"

namespace dojo::level06 {

// STUB — your job, and the keystone of this level.
//
// A histogram has every thread do atomicAdd(&bins[b], 1). When many lanes in a
// warp hit the SAME bin (common in real images — flat regions, skies), those
// atomics serialize: up to 32 round trips to the same address, one after
// another.
//
// The warp-aggregation trick collapses them:
//   1. Each lane computes its bin b.
//   2. __ballot_sync / __match_any_sync finds the set of lanes in this warp that
//      share lane's bin -> a peer mask.
//   3. The lowest-id lane in that group does ONE atomicAdd of __popc(mask)
//      (the group size) to bins[b]. The other lanes do nothing.
// 32 atomics to a hot bin become 1. This is the core of fast histograms and
// stream compaction.
//
// PREDICT before you profile: for an input where all 32 lanes of a warp map to
// the same bin, how many atomicAdds does the naive version issue vs. the
// aggregated one? What about a perfectly uniform input across many bins (where
// is aggregation's win smallest)?
//
// The host wrapper below is complete (it zeros the bins and copies results
// back). Write only the kernel body. Bin mapping: byte value v -> bin
// v * num_bins / 256.
__global__ void histogram_warp_kernel(const unsigned char *__restrict__ in,
                                      int n, unsigned int *__restrict__ bins,
                                      int num_bins) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < n; i += stride) {
    int b = static_cast<int>(in[i]) * num_bins / 256;
    // TODO(level06): increment bins[b]. Start with the correct-but-naive
    // atomicAdd(&bins[b], 1u), confirm it passes, then replace it with the
    // warp-aggregated version (ballot the peers sharing b, one lane adds the
    // group size). Measure both — the win shows up under contention.
    (void)b;
    (void)bins; // placeholder: nothing is counted yet, so the test fails
  }
}

void histogram_warp(const unsigned char *in, int n, unsigned int *bins,
                    int num_bins) {
  const std::size_t in_bytes = static_cast<std::size_t>(n);
  const std::size_t bin_bytes =
      static_cast<std::size_t>(num_bins) * sizeof(unsigned int);

  unsigned char *d_in = nullptr;
  unsigned int *d_bins = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, in_bytes));
  CUDA_CHECK(cudaMalloc(&d_bins, bin_bytes));

  CUDA_CHECK(cudaMemcpy(d_in, in, in_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_bins, 0, bin_bytes)); // bins start at zero

  constexpr int block = 256;
  const int grid = ceil_div(n, block) < 1024 ? ceil_div(n, block) : 1024;
  histogram_warp_kernel<<<grid, block>>>(d_in, n, d_bins, num_bins);
  CUDA_CHECK_KERNEL();

  CUDA_CHECK(cudaMemcpy(bins, d_bins, bin_bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_bins));
}

} // namespace dojo::level06
