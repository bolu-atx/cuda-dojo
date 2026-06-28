#include "level02.cuh"

#include "dojo/cuda_utils.cuh"

namespace dojo::level02 {

// STUB — your job, and the keystone of this level.
//
// A transpose moves element (x, y) of a w x h image to position (y, x) of the
// h x w output:
//
//     out[x*h + y] = in[y*w + x];
//
// Here is the trap that this whole level builds toward. A warp is 32 threads
// that differ in x (same y). Look at the two memory streams for that warp:
//
//   READ  in[y*w + x]  for x, x+1, ... x+31  -> 32 consecutive addresses
//                                               => ONE coalesced transaction.
//   WRITE out[x*h + y] for x, x+1, ... x+31  -> addresses h apart (stride h)
//                                               => 32 SEPARATE transactions.
//
// Swap the mapping so writes are coalesced and the *reads* become strided
// instead. You cannot make both ends coalesced with thread mapping alone — that
// is the cliffhanger. Level 4 fixes it by staging a tile in shared memory.
//
// Implement the naive version here, then PREDICT before profiling: roughly what
// fraction of peak bandwidth will a strided-on-one-end transpose reach?
__global__ void transpose_kernel(const float *in, float *out, int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < w && y < h) {
    // TODO(level02): write out[x*h + y] = in[y*w + x].
    (void)in;
    out[x * h + y] = 0.0f; // placeholder so the file compiles
  }
}

void transpose(const float *in, float *out, int w, int h) {
  const std::size_t bytes = static_cast<std::size_t>(w) * h * sizeof(float);

  float *d_in = nullptr;
  float *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));

  CUDA_CHECK(cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice));

  const dim3 block(32, 8);
  const dim3 grid(ceil_div(w, block.x), ceil_div(h, block.y));
  transpose_kernel<<<grid, block>>>(d_in, d_out, w, h);
  CUDA_CHECK_KERNEL();

  CUDA_CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
}

} // namespace dojo::level02
