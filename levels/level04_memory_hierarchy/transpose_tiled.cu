#include "level04.cuh"

#include "dojo/cuda_utils.cuh"

namespace dojo::level04 {

constexpr int kTile = 32;

// STUB — your job, and the keystone of this level.
//
// Plan (each block handles one 32x32 tile of the image):
//
//   1. READ phase — coalesced. Thread (tx, ty) reads input pixel
//      (blockIdx.x*32 + tx, blockIdx.y*32 + ty). Within a warp tx varies, so the
//      read is 32 consecutive floats => coalesced. Store it into the tile, but
//      transposed *in shared memory*:  tile[tx][ty] = in[...].
//
//   2. __syncthreads() — the whole tile must be loaded before anyone reads it
//      back in the other order. Why exactly here and not earlier?
//
//   3. WRITE phase — also coalesced. Compute the *transposed* output tile origin
//      (swap blockIdx.x and blockIdx.y) and write tile[ty][tx] so that, again,
//      consecutive threads in a warp write 32 consecutive output addresses.
//
// THE PADDING (the lesson): declare the tile as [kTile][kTile + 1]. Shared
// memory has 32 banks. A 32-wide tile makes a whole column fall in one bank, so
// the transposing access serializes 32-way. The +1 pad shifts each row by one
// bank, so a column now spreads across all 32 banks — no conflict.
__global__ void transpose_tiled_kernel(const float *in, float *out, int w,
                                       int h) {
  __shared__ float tile[kTile][kTile + 1]; // +1 pad: avoids 32-way bank conflict

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Input coordinate this thread reads (coalesced over tx).
  int in_x = blockIdx.x * kTile + tx;
  int in_y = blockIdx.y * kTile + ty;

  // TODO(level04): load into shared, transposed, with a bounds guard:
  //   if (in_x < w && in_y < h) tile[tx][ty] = in[in_y * w + in_x];
  (void)in;
  (void)tile;
  (void)in_x;
  (void)in_y;

  __syncthreads();

  // Output tile origin is the transpose of the input tile origin (swap block x/y).
  int out_x = blockIdx.y * kTile + tx; // column in the h-wide output
  int out_y = blockIdx.x * kTile + ty; // row in the w-tall output

  // TODO(level04): write back, coalesced, with a bounds guard against (h, w):
  //   if (out_x < h && out_y < w) out[out_y * h + out_x] = tile[ty][tx];
  if (out_x < h && out_y < w) {
    out[out_y * h + out_x] = 0.0f; // placeholder so the file compiles
  }
}

void transpose_tiled(const float *in, float *out, int w, int h) {
  const std::size_t bytes = static_cast<std::size_t>(w) * h * sizeof(float);

  float *d_in = nullptr;
  float *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));

  CUDA_CHECK(cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice));

  // One 32x32 block per tile; grid rounds up so edge tiles are partly out of
  // bounds (hence the guards in the kernel).
  const dim3 block(kTile, kTile);
  const dim3 grid(ceil_div(w, kTile), ceil_div(h, kTile));
  transpose_tiled_kernel<<<grid, block>>>(d_in, d_out, w, h);
  CUDA_CHECK_KERNEL();

  CUDA_CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
}

} // namespace dojo::level04
