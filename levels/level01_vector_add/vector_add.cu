#include "level01.cuh"

#include "dojo/cuda_utils.cuh"

namespace dojo::level01 {

// One thread per element. The grid-stride loop isn't strictly needed when we
// launch enough threads, but it's the idiomatic, launch-config-independent way
// to write an elementwise kernel and it's good to internalize early.
__global__ void vector_add_kernel(const float *a, const float *b, float *out,
                                  int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < n; i += stride) {
    out[i] = a[i] + b[i];
  }
}

void vector_add(const float *h_a, const float *h_b, float *h_out, int n) {
  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

  float *d_a = nullptr;
  float *d_b = nullptr;
  float *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));

  CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

  constexpr int block = 256;
  const int grid = ceil_div(n, block);
  vector_add_kernel<<<grid, block>>>(d_a, d_b, d_out, n);
  CUDA_CHECK_KERNEL();

  CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_out));
}

} // namespace dojo::level01
