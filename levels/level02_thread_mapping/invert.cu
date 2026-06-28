#include "level02.cuh"

#include "dojo/cuda_utils.cuh"

namespace dojo::level02 {

// COMPLETE worked example — the pattern every Level 2 kernel copies.
//
// We launch a 2D grid. threadIdx.x maps to the column (x), threadIdx.y to the
// row (y). Because consecutive threads differ in x, a warp accesses
// in[y*w + x], in[y*w + x+1], ... — 32 consecutive floats — which is one
// coalesced 128-byte transaction. That is the whole point of mapping x to the
// contiguous axis.
__global__ void invert_kernel(const float *in, float *out, int w, int h,
                              float maxval) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Two-sided guard: the grid is rounded up to whole blocks, so threads past the
  // right edge (x >= w) or bottom edge (y >= h) must not touch memory.
  if (x < w && y < h) {
    int idx = y * w + x;
    out[idx] = maxval - in[idx];
  }
}

void invert(const float *in, float *out, int w, int h, float maxval) {
  const std::size_t bytes = static_cast<std::size_t>(w) * h * sizeof(float);

  float *d_in = nullptr;
  float *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));

  CUDA_CHECK(cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice));

  // 32 lanes wide so a row of a warp is one coalesced transaction; 8 rows tall
  // keeps the block at 256 threads.
  const dim3 block(32, 8);
  const dim3 grid(ceil_div(w, block.x), ceil_div(h, block.y));
  invert_kernel<<<grid, block>>>(d_in, d_out, w, h, maxval);
  CUDA_CHECK_KERNEL();

  CUDA_CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
}

} // namespace dojo::level02
