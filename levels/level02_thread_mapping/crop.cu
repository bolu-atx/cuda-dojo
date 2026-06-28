#include "level02.cuh"

#include "dojo/cuda_utils.cuh"

namespace dojo::level02 {

// STUB — your job.
//
// Crop teaches the first real lesson of thread mapping: the output index is not
// the input index. Each thread owns one *output* pixel (x, y) in the
// outW x outH window, but it must read from the *source* pixel that lies at
// (x + x0, y + y0) in the larger inW x inH image.
//
// Questions to answer before you write a line:
//   - Which thread owns output pixel (x, y)? (derive the global 2D index)
//   - What is its address in `out`?   -> y * outW + x
//   - What is the address it reads from in `in`? (apply the (x0, y0) offset)
//   - The guard now uses the *output* dimensions. Why not the input dims?
__global__ void crop_kernel(const float *in, int inW, int /*inH*/, float *out,
                            int x0, int y0, int outW, int outH) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < outW && y < outH) {
    // TODO(level02): write out[y*outW + x] = in[ (y+y0)*inW + (x+x0) ].
    (void)in;
    (void)inW;
    (void)x0;
    (void)y0;
    out[y * outW + x] = 0.0f; // placeholder so the file compiles
  }
}

void crop(const float *in, int inW, int inH, float *out, int x0, int y0,
          int outW, int outH) {
  // Note the asymmetry: input and output have *different* sizes, so they need
  // separately sized allocations and copies. This is the first time the host
  // flow isn't a single uniform `bytes`.
  const std::size_t in_bytes = static_cast<std::size_t>(inW) * inH * sizeof(float);
  const std::size_t out_bytes = static_cast<std::size_t>(outW) * outH * sizeof(float);

  float *d_in = nullptr;
  float *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, in_bytes));
  CUDA_CHECK(cudaMalloc(&d_out, out_bytes));

  CUDA_CHECK(cudaMemcpy(d_in, in, in_bytes, cudaMemcpyHostToDevice));

  const dim3 block(32, 8);
  const dim3 grid(ceil_div(outW, block.x), ceil_div(outH, block.y));
  crop_kernel<<<grid, block>>>(d_in, inW, inH, d_out, x0, y0, outW, outH);
  CUDA_CHECK_KERNEL();

  CUDA_CHECK(cudaMemcpy(out, d_out, out_bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
}

} // namespace dojo::level02
