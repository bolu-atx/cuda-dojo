#include "level05.cuh"

#include "dojo/cuda_utils.cuh"

namespace dojo::level05 {

// STUB — your job. Same numeric result as box_blur_naive, but faster.
//
// First measure the baseline with the demo and `ncu --set full`. You should find
// it memory-bound and well below peak bandwidth — the reuse is being thrown away:
// adjacent output pixels share almost their entire window, yet each thread
// re-fetches all (2r+1)^2 inputs from global.
//
// Pick ONE optimization and predict its effect before you run:
//
//   (a) Shared-memory tile. Each block cooperatively loads its pixels plus a
//       `radius`-wide halo into __shared__ once, then every thread reads the
//       window from shared (~25 cyc) instead of global (~600). Reuse factor goes
//       from "hope L1 catches it" to explicit.
//
//   (b) Separable passes. A box blur is separable: blur horizontally, then
//       vertically. That turns (2r+1)^2 reads per pixel into 2*(2r+1) — the
//       arithmetic-intensity denominator shrinks, so you move far fewer bytes.
//
// Target from the docs: >= 80% of peak bandwidth on this memory-bound op. After
// you implement it, report achieved GB/s before and after and state which of the
// four memory sins you removed.
__global__ void box_blur_opt_kernel(const float *in, float *out, int w, int h,
                                    int radius) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < w && y < h) {
    // TODO(level05): produce the same mean as box_blur_naive, but exploit reuse
    // (shared tile + halo) or separability so you move far fewer bytes.
    (void)in;
    (void)radius;
    out[y * w + x] = 0.0f; // placeholder so the file compiles
  }
}

void box_blur_opt(const float *in, float *out, int w, int h, int radius) {
  const std::size_t bytes = static_cast<std::size_t>(w) * h * sizeof(float);

  float *d_in = nullptr;
  float *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));

  CUDA_CHECK(cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice));

  // If you choose a shared-tile approach you'll likely want a square block so the
  // halo is symmetric; left as-is for now.
  const dim3 block(32, 8);
  const dim3 grid(ceil_div(w, block.x), ceil_div(h, block.y));
  box_blur_opt_kernel<<<grid, block>>>(d_in, d_out, w, h, radius);
  CUDA_CHECK_KERNEL();

  CUDA_CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
}

} // namespace dojo::level05
