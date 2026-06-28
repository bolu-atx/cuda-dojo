#include "dojo/cuda_utils.cuh"

#include <cstdio>
#include <vector>

// Level 5 STRETCH demo — padding vs. swizzling for bank-conflict-free shared
// memory. Not a graded exercise (no test, not in level05.cuh); it's a pure
// profiling target that proves the doc's claim. All three kernels transpose a
// square matrix through a shared tile; they differ ONLY in how that tile is laid
// out, which is the whole lesson.
//
// The Level 5 worked example (box filter) killed bank conflicts by PADDING the
// inner dimension: tile[N][N+1]. The transpose stub at Level 3 did the same. That
// works, but it costs you. Swizzling is the other tool: leave the tile a clean
// power-of-two square and instead PERMUTE the column index with XOR so a warp's
// 32 lanes still scatter across all 32 banks.
//
// Why swizzle ever beats padding (the cliffhanger, paid off at Level 9 GEMM):
//   1. Padding wastes shared memory. tile[32][33] burns 1/33 of the tile for
//      nothing. Shared memory is the scarce resource that caps occupancy, so on a
//      smem-bound kernel padding directly lowers how many blocks fit per SM.
//   2. Padding breaks the power-of-two row stride that 128-bit vectorized
//      ld.shared/st.shared needs. A stride of 33 floats is unaligned for float4
//      access; a stride of 32 is not. CUTLASS-style GEMM cannot give up vectorized
//      shared loads, so it swizzles. On a plain transpose neither cost bites yet —
//      which is exactly why this is a stretch, not the main lesson.
//
// THE SWIZZLE. Shared memory has 32 banks; bank(addr) = (addr / 4) % 32. For a
// 32-wide tile, element (row, col) lands in bank `col % 32` = col. A transpose
// reads a whole COLUMN (col fixed, row sweeps 0..31) — every lane hits bank `col`,
// a 32-way conflict. Swizzle the stored column to `col ^ row`:
//   - transpose read (col fixed, row = 0..31): banks are {col ^ 0, ..., col ^ 31}
//     = a permutation of 0..31  -> all distinct -> no conflict.
//   - tile load     (row fixed, col = 0..31): banks are {0 ^ row, ..., 31 ^ row}
//     = also a permutation of 0..31  -> no conflict.
// XOR is its own inverse and a bijection per fixed operand, so it is conflict-free
// in BOTH directions with zero wasted memory. Same bank-conflict model as Level
// 3/4, different tool.
//
// PREDICT before you profile: how many shared bank conflicts (per warp) does each
// kernel hit on the transposing access — naive, padded, swizzled? Then check:
//   ncu --set full ./level05_swizzle_demo
//   metric: "Shared Memory Bank Conflicts" (l1tex__data_bank_conflicts_pipe_lsu_*)
//   naive spikes (~31 way), padded ~0, swizzled ~0. Confirm, don't assume.

namespace {

constexpr int kTile = 32; // one warp wide on purpose: matches the 32 banks.

// Baseline: clean square tile, NO pad, NO swizzle. The transposing read is a
// pure column access -> 32-way bank conflict. This is the wall the other two fix.
__global__ void transpose_naive_smem(const float *__restrict__ in,
                                     float *__restrict__ out, int w, int h) {
  __shared__ float tile[kTile][kTile];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int in_x = blockIdx.x * kTile + tx;
  const int in_y = blockIdx.y * kTile + ty;

  if (in_x < w && in_y < h)
    tile[ty][tx] = in[in_y * w + in_x]; // coalesced load (conflict-free write)
  __syncthreads();

  // Output tile origin is the transpose of the input tile origin.
  const int out_x = blockIdx.y * kTile + tx;
  const int out_y = blockIdx.x * kTile + ty;
  if (out_x < h && out_y < w)
    out[out_y * h + out_x] = tile[tx][ty]; // COLUMN read -> 32-way conflict
}

// The Level 5 / Level 3 fix: +1 pad skews each row by one bank so a column read
// spreads across all 32 banks. Conflict-free, but stride is 33 (see header).
__global__ void transpose_padded(const float *__restrict__ in,
                                 float *__restrict__ out, int w, int h) {
  __shared__ float tile[kTile][kTile + 1];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int in_x = blockIdx.x * kTile + tx;
  const int in_y = blockIdx.y * kTile + ty;

  if (in_x < w && in_y < h)
    tile[ty][tx] = in[in_y * w + in_x];
  __syncthreads();

  const int out_x = blockIdx.y * kTile + tx;
  const int out_y = blockIdx.x * kTile + ty;
  if (out_x < h && out_y < w)
    out[out_y * h + out_x] = tile[tx][ty];
}

// The swizzle: clean 32x32 tile, store column at (col ^ row). Same conflict-free
// result as padding with no wasted memory and a power-of-two stride. Note the XOR
// must be applied identically on store and load so we read back the same cell.
__global__ void transpose_swizzled(const float *__restrict__ in,
                                   float *__restrict__ out, int w, int h) {
  __shared__ float tile[kTile][kTile];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int in_x = blockIdx.x * kTile + tx;
  const int in_y = blockIdx.y * kTile + ty;

  if (in_x < w && in_y < h)
    tile[ty][tx ^ ty] = in[in_y * w + in_x]; // store col swizzled by row
  __syncthreads();

  const int out_x = blockIdx.y * kTile + tx;
  const int out_y = blockIdx.x * kTile + ty;
  if (out_x < h && out_y < w)
    out[out_y * h + out_x] = tile[tx][ty ^ tx]; // read it back the same way
}

void cpu_transpose(const std::vector<float> &in, std::vector<float> &out, int w,
                   int h) {
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      out[static_cast<std::size_t>(x) * h + y] =
          in[static_cast<std::size_t>(y) * w + x];
}

// Launch one kernel end-to-end (alloc/copy/launch/copy back), verify against the
// CPU reference, and report time + effective bandwidth.
template <typename Kernel>
void run(const char *name, Kernel kernel, const std::vector<float> &in,
         const std::vector<float> &ref, int w, int h) {
  const std::size_t bytes = static_cast<std::size_t>(w) * h * sizeof(float);
  std::vector<float> out(ref.size());

  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_in, in.data(), bytes, cudaMemcpyHostToDevice));

  const dim3 block(kTile, kTile);
  const dim3 grid(ceil_div(w, kTile), ceil_div(h, kTile));

  dojo::GpuTimer t;
  t.start();
  kernel<<<grid, block>>>(d_in, d_out, w, h);
  const float ms = t.stop();
  CUDA_CHECK_KERNEL();

  CUDA_CHECK(cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));

  bool ok = true;
  for (std::size_t i = 0; i < ref.size() && ok; ++i)
    ok = out[i] == ref[i];

  // Transpose moves the matrix once in, once out: ~2 passes of traffic.
  const std::size_t two_passes = 2ull * bytes;
  std::printf("%-20s %s  %.3f ms, ~%.1f GB/s\n", name, ok ? "PASS" : "FAIL", ms,
              dojo::gbps(two_passes, ms));
}

} // namespace

int main() {
  dojo::print_device_info();

  // A clean multiple of 32 keeps every tile full, so the profiler reads the bank
  // behavior of the access pattern itself, not edge-tile noise. The guards still
  // protect ragged sizes.
  const int w = 4096, h = 4096;
  std::printf("\ntranspose %dx%d (%.1f MB) through a 32x32 shared tile\n", w, h,
              static_cast<double>(w) * h * sizeof(float) / (1024.0 * 1024.0));
  std::printf("predict the bank conflicts per warp, then run under ncu:\n");
  std::printf("  ncu --set full ./level05_swizzle_demo\n\n");

  std::vector<float> in(static_cast<std::size_t>(w) * h), ref(in.size());
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      in[static_cast<std::size_t>(y) * w + x] =
          static_cast<float>((x * 7 + y * 13) % 251);
  cpu_transpose(in, ref, w, h);

  run("naive (no pad)", transpose_naive_smem, in, ref, w, h);
  run("padded [32][33]", transpose_padded, in, ref, w, h);
  run("swizzled col^row", transpose_swizzled, in, ref, w, h);

  return 0;
}
