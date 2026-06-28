#include "level05.cuh"

#include "dojo/cuda_utils.cuh"

namespace dojo::level05 {

// STUB — your job. Same numeric result as box_filter_shared, but cheaper.
//
// A box blur is SEPARABLE: blurring a (2r+1)x(2r+1) window equals a horizontal
// 1D blur of width (2r+1) followed by a vertical one. The win is in the read
// count per output pixel:
//
//     2D box        : (2r+1)^2  reads      (r=4 -> 81)
//     separable     : 2*(2r+1)  reads      (r=4 -> 18)
//
// You move far fewer bytes, so on this memory-bound op you should land much
// closer to peak bandwidth. PREDICT before you profile: at r=4, what fraction of
// the 2D version's DRAM traffic does the separable version move? (~18/81.)
//
// The two passes below each want the same Level 5 rhythm, but with a 1D tile:
// a block loads a row strip (pass 1) or a column strip (pass 2) plus an r-wide
// halo into __shared__, __syncthreads(), then averages along that one axis from
// shared. Clamp coordinates to the border in the load, exactly like the worked
// example, so the two clamped 1D passes compose to the clamped 2D box.
//
// Both kernels divide by (2r+1) (NOT (2r+1)^2): each pass is a 1D mean, and the
// two means multiply to the 2D mean.

constexpr int kBlock = 16;

// Pass 1: horizontal mean. out_tmp[y,x] = mean(in[y, x-r .. x+r]) (clamped).
__global__ void blur_rows_kernel(const float *__restrict__ in,
                                 float *__restrict__ tmp, int w, int h,
                                 int radius) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < w && y < h) {
    // TODO(level05): average in[y, x-radius .. x+radius] with clamped columns
    // and divide by (2*radius+1). Stage the row strip + halo in __shared__ and
    // __syncthreads() before reading it back.
    (void)in;
    (void)radius;
    tmp[y * w + x] = 0.0f; // placeholder so the file compiles
  }
}

// Pass 2: vertical mean. out[y,x] = mean(tmp[y-r .. y+r, x]) (clamped).
__global__ void blur_cols_kernel(const float *__restrict__ tmp,
                                 float *__restrict__ out, int w, int h,
                                 int radius) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < w && y < h) {
    // TODO(level05): average tmp[y-radius .. y+radius, x] with clamped rows and
    // divide by (2*radius+1). Same shared-tile rhythm, now along y.
    (void)tmp;
    (void)radius;
    out[y * w + x] = 0.0f; // placeholder so the file compiles
  }
}

void separable_blur(const float *in, float *out, int w, int h, int radius) {
  const std::size_t bytes = static_cast<std::size_t>(w) * h * sizeof(float);

  float *d_in = nullptr;
  float *d_tmp = nullptr; // horizontal-pass result, fed into the vertical pass
  float *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_tmp, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));

  CUDA_CHECK(cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice));

  const dim3 block(kBlock, kBlock);
  const dim3 grid(ceil_div(w, kBlock), ceil_div(h, kBlock));

  // Two launches, in order: the vertical pass reads what the horizontal pass
  // wrote. The kernel launches on the same (default) stream run sequentially, so
  // no explicit sync is needed between them.
  blur_rows_kernel<<<grid, block>>>(d_in, d_tmp, w, h, radius);
  CUDA_CHECK_KERNEL();
  blur_cols_kernel<<<grid, block>>>(d_tmp, d_out, w, h, radius);
  CUDA_CHECK_KERNEL();

  CUDA_CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_tmp));
  CUDA_CHECK(cudaFree(d_out));
}

} // namespace dojo::level05
