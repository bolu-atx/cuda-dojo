#include "level01.cuh"

#include "dojo/cuda_utils.cuh"

namespace dojo::level01 {

// out[i] = alpha * x[i] + y[i]. `alpha` is passed by value: scalar kernel
// arguments are copied into constant/parameter memory on launch — no need to
// allocate device memory for it.
__global__ void saxpy_kernel(float alpha, const float *x, const float *y,
                             float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < n; i += stride) {
    out[i] = alpha * x[i] + y[i]; // FMA: a single fused multiply-add per element
  }
}

void saxpy(float alpha, const float *h_x, const float *h_y, float *h_out,
           int n) {
  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

  float *d_x = nullptr;
  float *d_y = nullptr;
  float *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, bytes));
  CUDA_CHECK(cudaMalloc(&d_y, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));

  CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

  constexpr int block = 256;
  const int grid = ceil_div(n, block);
  saxpy_kernel<<<grid, block>>>(alpha, d_x, d_y, d_out, n);
  CUDA_CHECK_KERNEL();

  CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_out));
}

} // namespace dojo::level01
