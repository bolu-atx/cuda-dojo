#include <cstdio>

namespace {

int warp_id(int thread_idx_x) { return thread_idx_x >> 5; }
int lane_id(int thread_idx_x) { return thread_idx_x & 31; }

} // namespace

int main() {
  std::puts("Level 3 — logical vs physical execution");
  std::puts("For a one-dimensional block, warp = threadIdx.x / 32 and lane = threadIdx.x % 32.");
  std::puts("");
  std::puts("threadIdx.x  warp  lane");
  for (int tid : {0, 31, 32, 63, 64, 70, 127, 255}) {
    std::printf("%11d  %4d  %4d\n", tid, warp_id(tid), lane_id(tid));
  }
  std::puts("");
  std::puts("The programmer launches blocks. The runtime schedules those blocks onto SMs.");
  std::puts("A block never spans multiple SMs, so shared memory and __syncthreads() are block-scoped.");
  std::puts("");
  std::puts("Can these two threads cooperate directly, inside one kernel?");
  std::puts("  same warp .................. yes  -> registers via __shfl_*_sync");
  std::puts("  same block, other warp ..... yes  -> __shared__ + __syncthreads()");
  std::puts("  different block ............ no   -> needs a kernel boundary / global protocol");
  std::puts("");
  std::puts("Reason it through: which row changes if you grow the grid? (None -- the");
  std::puts("cooperation boundary is physical, set by the SM, not by how you index.)");
  return 0;
}
