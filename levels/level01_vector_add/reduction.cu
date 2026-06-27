#include "level01.cuh"

#include "dojo/cuda_utils.cuh"

#include <vector>

namespace dojo::level01 {

// A first, deliberately simple reduction. Each block sums a grid-stride slice
// of the input into shared memory, tree-reduces within the block, and writes
// one partial sum to global memory. The host adds up the (small number of)
// partials. We'll revisit reduction properly at Level 6 with warp shuffles —
// this version exists so you can see shared memory + __syncthreads() once.
constexpr int kBlock = 256;

__global__ void reduce_sum_kernel(const float *in, float *partials, int n) {
  __shared__ float sdata[kBlock];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // Each thread accumulates a private running sum over its grid-stride slice
  // before we ever touch shared memory — fewer __syncthreads(), more work per
  // thread.
  float local = 0.0f;
  for (int i = idx; i < n; i += stride) {
    local += in[i];
  }
  sdata[tid] = local;
  __syncthreads();

  // Tree reduction in shared memory: 256 -> 128 -> ... -> 1.
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partials[blockIdx.x] = sdata[0];
  }
}

float reduce_sum(const float *h_in, int n) {
  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

  // Cap the grid so each block does a healthy amount of work and the host-side
  // final sum stays cheap.
  const int grid = ceil_div(n, kBlock) < 1024 ? ceil_div(n, kBlock) : 1024;

  float *d_in = nullptr;
  float *d_partials = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_partials, static_cast<std::size_t>(grid) * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

  reduce_sum_kernel<<<grid, kBlock>>>(d_in, d_partials, n);
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

} // namespace dojo::level01
