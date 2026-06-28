#include "level05.cuh"

#include "dojo/cuda_utils.cuh"

namespace dojo::level05 {

// COMPLETE. Sobel gradient magnitude over the 3x3 neighborhood, edges clamped.
//
//   Gx = [-1 0 1; -2 0 2; -1 0 1] * window
//   Gy = [-1 -2 -1; 0 0 0; 1 2 1] * window
//   out = sqrt(Gx*Gx + Gy*Gy)
//
// Use this to practice the Level 5 habit: compute arithmetic intensity first.
// Per output pixel it reads 9 floats (36 bytes) + writes 4 bytes, and does ~20
// FLOPs (two weighted sums, two squares, an add, a sqrt). AI ~= 20/40 = 0.5
// FLOP/byte => deeply memory-bound, same regime as the blur. The sqrt does not
// change that classification — which is exactly the point.
__global__ void sobel_kernel(const float *in, float *out, int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < w && y < h) {
    auto at = [&](int dx, int dy) -> float {
      int nx = min(max(x + dx, 0), w - 1);
      int ny = min(max(y + dy, 0), h - 1);
      return in[ny * w + nx];
    };

    float gx = -at(-1, -1) + at(1, -1) - 2.0f * at(-1, 0) + 2.0f * at(1, 0) -
               at(-1, 1) + at(1, 1);
    float gy = -at(-1, -1) - 2.0f * at(0, -1) - at(1, -1) + at(-1, 1) +
               2.0f * at(0, 1) + at(1, 1);

    out[y * w + x] = sqrtf(gx * gx + gy * gy);
  }
}

void sobel(const float *in, float *out, int w, int h) {
  const std::size_t bytes = static_cast<std::size_t>(w) * h * sizeof(float);

  float *d_in = nullptr;
  float *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));

  CUDA_CHECK(cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice));

  const dim3 block(32, 8);
  const dim3 grid(ceil_div(w, block.x), ceil_div(h, block.y));
  sobel_kernel<<<grid, block>>>(d_in, d_out, w, h);
  CUDA_CHECK_KERNEL();

  CUDA_CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
}

} // namespace dojo::level05
