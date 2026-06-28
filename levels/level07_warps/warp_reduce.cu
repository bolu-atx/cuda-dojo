#include "level07.cuh"

#include "dojo/cuda_utils.cuh"

#include <vector>

namespace dojo::level07 {

// COMPLETE worked example — the warp-reduce pattern every Level 7 kernel copies.
//
// Compare this against Level 1's reduce_sum, which tree-reduced 256 values in
// shared memory under log2(256) = 8 __syncthreads(). Here the heavy lifting
// happens in registers via __shfl_down_sync, and the block needs exactly ONE
// barrier.
constexpr int kBlock = 256;          // 8 warps per block
constexpr int kWarps = kBlock / 32;  // partials we stage in shared

// A full-warp sum in five shuffles. After step `offset`, lane i holds the sum of
// lanes [i, i+offset). No __syncthreads(): the 32 lanes are one SIMD
// instruction, so each lane sees the others' registers the instant the shuffle
// retires. Lane 0 ends up with the warp's total (other lanes hold partials we
// ignore). Full warp here, so the mask is 0xffffffff.
__device__ float warp_reduce_sum(float v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

__global__ void reduce_sum_warp_kernel(const float *__restrict__ in,
                                       float *__restrict__ partials, int n) {
  // One slot per warp for its partial sum — the only shared memory we need.
  __shared__ float warp_sums[kWarps];

  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;
  const int stride = blockDim.x * gridDim.x;
  const int lane = tid & 31; // lane index within the warp (tid % 32)
  const int warp = tid >> 5; // warp index within the block (tid / 32)

  // Each thread accumulates a private running sum over its grid-stride slice.
  // Threads with idx >= n simply add nothing and keep local = 0 — every lane
  // still participates in the shuffles below, so the warp never diverges and the
  // 0xffffffff mask stays valid.
  float local = 0.0f;
  for (int i = idx; i < n; i += stride) {
    local += in[i];
  }

  // Stage 1: reduce within each warp, barrier-free. Lane 0 of each warp now
  // holds that warp's sum; write it to shared.
  local = warp_reduce_sum(local);
  if (lane == 0) {
    warp_sums[warp] = local;
  }

  // The ONLY barrier: warp 0 is about to read partials that other warps wrote.
  __syncthreads();

  // Stage 2: warp 0 reduces the (kWarps) partials with the same shuffle tree.
  // Lanes past kWarps load the identity (0) so the full-warp shuffle is valid.
  if (warp == 0) {
    float v = (lane < kWarps) ? warp_sums[lane] : 0.0f;
    v = warp_reduce_sum(v);
    if (lane == 0) {
      partials[blockIdx.x] = v;
    }
  }
}

float reduce_sum_warp(const float *h_in, int n) {
  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

  // Cap the grid so each block does real work and the host-side final add over
  // partials stays cheap (same shape as Level 1's reduction).
  const int grid = ceil_div(n, kBlock) < 1024 ? ceil_div(n, kBlock) : 1024;

  float *d_in = nullptr;
  float *d_partials = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_partials,
                        static_cast<std::size_t>(grid) * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

  reduce_sum_warp_kernel<<<grid, kBlock>>>(d_in, d_partials, n);
  CUDA_CHECK_KERNEL();

  std::vector<float> partials(grid);
  CUDA_CHECK(cudaMemcpy(partials.data(), d_partials,
                        static_cast<std::size_t>(grid) * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_partials));

  double total = 0.0; // accumulate in double to limit summation error
  for (float p : partials) {
    total += p;
  }
  return static_cast<float>(total);
}

} // namespace dojo::level07
