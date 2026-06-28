#include "level04.cuh"

#include "dojo/cuda_utils.cuh"

namespace dojo::level04 {

// COMPLETE baseline — carried over from Level 2 so this level is self-contained
// and the demo can A/B it against the tiled version.
//
// Read in[y*w + x] is coalesced (a warp reads 32 consecutive floats). Write
// out[x*h + y] is strided by h (a warp writes 32 addresses h apart) => many
// transactions. This is the wall thread mapping alone cannot climb.
__global__ void transpose_naive_kernel(const float *in, float *out, int w,
                                       int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < w && y < h) {
    out[x * h + y] = in[y * w + x];
  }
}

void transpose_naive(const float *in, float *out, int w, int h) {
  const std::size_t bytes = static_cast<std::size_t>(w) * h * sizeof(float);

  float *d_in = nullptr;
  float *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));

  CUDA_CHECK(cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice));

  const dim3 block(32, 8);
  const dim3 grid(ceil_div(w, block.x), ceil_div(h, block.y));
  transpose_naive_kernel<<<grid, block>>>(d_in, d_out, w, h);
  CUDA_CHECK_KERNEL();

  CUDA_CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
}

} // namespace dojo::level04
