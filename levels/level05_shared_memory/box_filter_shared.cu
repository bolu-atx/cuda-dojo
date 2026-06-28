#include "level05.cuh"

#include "dojo/cuda_utils.cuh"

namespace dojo::level05 {

// COMPLETE worked example — the pattern every Level 5 kernel copies.
//
// A (2r+1)x(2r+1) box blur, edges clamped. The naive version (Level 3's
// blur3x3_global) sent every one of the (2r+1)^2 neighbor reads straight to
// global and prayed the L1/L2 caught the reuse. Here we make the reuse explicit:
// each block stages its 16x16 output tile PLUS an r-wide halo into __shared__
// exactly once, then every thread averages its window out of fast on-chip
// memory.
//
// Geometry justification:
//   - kBlock = 16 -> 256 threads/block (a healthy occupancy default) and a
//     square tile so the halo is symmetric in x and y.
//   - The shared tile is (16 + 2r) on a side. We size the static array for the
//     worst case (kMaxRadius) and require radius <= kMaxRadius at launch.
//   - +1 pad on the inner dimension (the bank-conflict lesson from Level 4):
//     shared memory has 32 banks, so an unpadded power-of-two row stride makes a
//     column access collide. The pad skews each row by one bank.
//
// The reuse factor: this block produces up to 16*16 = 256 outputs, each reading
// (2r+1)^2 pixels = 256*(2r+1)^2 logical reads, but it fetches only (16+2r)^2
// pixels from global. For r=1 that's 2304 naive reads collapsed into 324 fetches
// — ~7x less DRAM traffic, and that ratio is your speedup ceiling.
constexpr int kBlock = 16;
constexpr int kMaxRadius = 8;
constexpr int kTileDim = kBlock + 2 * kMaxRadius; // max side of the shared tile

__global__ void box_filter_shared_kernel(const float *__restrict__ in,
                                         float *__restrict__ out, int w, int h,
                                         int radius) {
  // +1 pad on the inner dimension to break 32-way bank conflicts on column reads.
  __shared__ float tile[kTileDim][kTileDim + 1];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * kBlock + tx; // 0..255, a flat id for the cooperative load

  const int tileDim = kBlock + 2 * radius; // actual side for this radius
  // Top-left image coordinate covered by this block's shared tile (halo
  // included) — note the -radius shift so the halo lives at tile index 0.
  const int originX = blockIdx.x * kBlock - radius;
  const int originY = blockIdx.y * kBlock - radius;

  // LOAD phase. There are tileDim*tileDim shared cells but only 256 threads, so
  // each thread grabs several cells in a strided sweep. Out-of-image samples are
  // clamped to the nearest border pixel, which is exactly the edge rule the CPU
  // reference uses — so the halo already holds replicated border values and the
  // compute loop never needs its own clamp.
  for (int i = tid; i < tileDim * tileDim; i += kBlock * kBlock) {
    const int ly = i / tileDim;
    const int lx = i % tileDim;
    int gx = min(max(originX + lx, 0), w - 1);
    int gy = min(max(originY + ly, 0), h - 1);
    tile[ly][lx] = in[gy * w + gx];
  }

  // BARRIER. Reached unconditionally by all 256 threads — the load guards are
  // on memory *addresses*, not on whether a thread participates. Putting this
  // inside the `if (x < w ...)` compute guard below would deadlock the block on
  // edge tiles where some threads return early. Why here? Because the compute
  // phase reads cells its neighbors loaded; nobody may read before everyone has
  // written.
  __syncthreads();

  const int x = blockIdx.x * kBlock + tx;
  const int y = blockIdx.y * kBlock + ty;
  if (x < w && y < h) {
    // This thread's center sits at (tx+radius, ty+radius) in tile coordinates;
    // the window [-radius, radius] therefore stays within [0, tileDim).
    float sum = 0.0f;
    for (int dy = -radius; dy <= radius; ++dy) {
      for (int dx = -radius; dx <= radius; ++dx) {
        sum += tile[ty + radius + dy][tx + radius + dx];
      }
    }
    const float count = static_cast<float>((2 * radius + 1) * (2 * radius + 1));
    out[y * w + x] = sum / count;
  }
}

void box_filter_shared(const float *in, float *out, int w, int h, int radius) {
  const std::size_t bytes = static_cast<std::size_t>(w) * h * sizeof(float);

  float *d_in = nullptr;
  float *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));

  CUDA_CHECK(cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice));

  // One 16x16 block per output tile; grid rounds up so edge tiles run partly out
  // of bounds (hence the compute guard and the clamped halo load).
  const dim3 block(kBlock, kBlock);
  const dim3 grid(ceil_div(w, kBlock), ceil_div(h, kBlock));
  box_filter_shared_kernel<<<grid, block>>>(d_in, d_out, w, h, radius);
  CUDA_CHECK_KERNEL();

  CUDA_CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
}

} // namespace dojo::level05
