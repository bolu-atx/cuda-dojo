#include "level04.cuh"

#include "dojo/cuda_utils.cuh"

namespace dojo::level04 {

// The 3x3 filter weights live in constant memory: 64 KB, cached, and broadcast
// at register speed when every thread in a warp reads the *same* address — which
// is exactly how a stencil reads its weights. The host fills this once via
// cudaMemcpyToSymbol before launching.
__constant__ float c_filter[9];

// STUB — your job.
//
// Same 3x3 blur as blur3x3_global, but with two read-path upgrades to exploit
// the reuse factor instead of leaning on luck:
//
//   1. Weights come from c_filter (constant memory). Because all threads read
//      c_filter[k] for the same k at the same time, the constant cache broadcasts
//      it — no global traffic for the weights at all.
//
//   2. The input pointer is `const float* __restrict__`. __restrict__ promises
//      the compiler `in` and `out` never alias, which lets it route the loads
//      through the read-only data cache (the same path as __ldg). For a stencil
//      that re-reads neighbors, that cache is what turns the reuse into hits.
//
// Implement the 3x3 weighted sum using c_filter[dy*3 + dx + ...] and the same
// border clamping as the baseline. For a box blur every weight is 1/9, so the
// result must match blur3x3_global to tolerance.
__global__ void blur3x3_constant_kernel(const float *__restrict__ in,
                                        float *__restrict__ out, int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < w && y < h) {
    // TODO(level04): accumulate sum += c_filter[(dy+1)*3 + (dx+1)] * in[clamped]
    //                over dy,dx in [-1,1], then out[y*w + x] = sum.
    (void)in;
    out[y * w + x] = 0.0f; // placeholder so the file compiles
  }
}

void blur3x3_constant(const float *in, float *out, int w, int h) {
  const std::size_t bytes = static_cast<std::size_t>(w) * h * sizeof(float);

  // Box-blur weights: every tap is 1/9. Upload to the __constant__ symbol once.
  float h_filter[9];
  for (int i = 0; i < 9; ++i) {
    h_filter[i] = 1.0f / 9.0f;
  }
  CUDA_CHECK(cudaMemcpyToSymbol(c_filter, h_filter, sizeof(h_filter)));

  float *d_in = nullptr;
  float *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));

  CUDA_CHECK(cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice));

  const dim3 block(32, 8);
  const dim3 grid(ceil_div(w, block.x), ceil_div(h, block.y));
  blur3x3_constant_kernel<<<grid, block>>>(d_in, d_out, w, h);
  CUDA_CHECK_KERNEL();

  CUDA_CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
}

} // namespace dojo::level04
