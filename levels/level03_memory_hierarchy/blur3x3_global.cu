#include "level03.cuh"

#include "dojo/cuda_utils.cuh"

namespace dojo::level03 {

// COMPLETE baseline — a 3x3 box blur (mean of the 9-pixel neighborhood) with
// edges clamped (out-of-range neighbors reuse the nearest valid pixel).
//
// Reuse factor lives here: a single input pixel is read by up to 9 different
// output threads (its own and its 8 neighbors'). This version makes every one of
// those reads go to global memory independently — it leans on the L1/L2 caches
// to absorb the reuse rather than staging it itself. The constant/read-only
// variant (and, at Level 5, a shared tile) make that reuse explicit.
__global__ void blur3x3_global_kernel(const float *in, float *out, int w,
                                      int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < w && y < h) {
    float sum = 0.0f;
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        // Clamp neighbor coordinates to the image border.
        int nx = min(max(x + dx, 0), w - 1);
        int ny = min(max(y + dy, 0), h - 1);
        sum += in[ny * w + nx];
      }
    }
    out[y * w + x] = sum / 9.0f;
  }
}

void blur3x3_global(const float *in, float *out, int w, int h) {
  const std::size_t bytes = static_cast<std::size_t>(w) * h * sizeof(float);

  float *d_in = nullptr;
  float *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));

  CUDA_CHECK(cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice));

  const dim3 block(32, 8);
  const dim3 grid(ceil_div(w, block.x), ceil_div(h, block.y));
  blur3x3_global_kernel<<<grid, block>>>(d_in, d_out, w, h);
  CUDA_CHECK_KERNEL();

  CUDA_CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
}

} // namespace dojo::level03
