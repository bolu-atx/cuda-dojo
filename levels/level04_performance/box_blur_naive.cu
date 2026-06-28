#include "level04.cuh"

#include "dojo/cuda_utils.cuh"

namespace dojo::level04 {

// COMPLETE baseline. Box blur of arbitrary radius: each output averages a
// (2r+1)x(2r+1) window, edges clamped. Every thread re-reads its whole window
// straight from global memory.
//
// Arithmetic intensity check (do this before optimizing): per output pixel this
// moves ~4 bytes out and reads (2r+1)^2 inputs, doing ~(2r+1)^2 adds + 1 divide.
// That is roughly 1 FLOP per byte read => firmly memory-bound. No math trick will
// help; the win is in how the bytes are fetched (coalescing + reuse).
__global__ void box_blur_naive_kernel(const float *in, float *out, int w, int h,
                                      int radius) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < w && y < h) {
    float sum = 0.0f;
    int count = 0;
    for (int dy = -radius; dy <= radius; ++dy) {
      for (int dx = -radius; dx <= radius; ++dx) {
        int nx = min(max(x + dx, 0), w - 1);
        int ny = min(max(y + dy, 0), h - 1);
        sum += in[ny * w + nx];
        ++count;
      }
    }
    out[y * w + x] = sum / static_cast<float>(count);
  }
}

void box_blur_naive(const float *in, float *out, int w, int h, int radius) {
  const std::size_t bytes = static_cast<std::size_t>(w) * h * sizeof(float);

  float *d_in = nullptr;
  float *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));

  CUDA_CHECK(cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice));

  const dim3 block(32, 8);
  const dim3 grid(ceil_div(w, block.x), ceil_div(h, block.y));
  box_blur_naive_kernel<<<grid, block>>>(d_in, d_out, w, h, radius);
  CUDA_CHECK_KERNEL();

  CUDA_CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
}

} // namespace dojo::level04
