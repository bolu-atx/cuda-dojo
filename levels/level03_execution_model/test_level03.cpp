#include "dojo/test.hpp"

namespace {

int warp_id(int thread_idx_x) { return thread_idx_x >> 5; }
int lane_id(int thread_idx_x) { return thread_idx_x & 31; }

} // namespace

DOJO_TEST_CASE(warp_and_lane_ids_match_cuda_convention) {
  DOJO_CHECK(warp_id(0) == 0);
  DOJO_CHECK(lane_id(0) == 0);

  DOJO_CHECK(warp_id(31) == 0);
  DOJO_CHECK(lane_id(31) == 31);

  DOJO_CHECK(warp_id(32) == 1);
  DOJO_CHECK(lane_id(32) == 0);

  DOJO_CHECK(warp_id(70) == 2);
  DOJO_CHECK(lane_id(70) == 6);
}

DOJO_TEST_CASE(block_size_implies_warp_count_rounding) {
  auto warps_for_block = [](int block_dim_x) {
    return (block_dim_x + 31) / 32;
  };

  DOJO_CHECK(warps_for_block(1) == 1);
  DOJO_CHECK(warps_for_block(32) == 1);
  DOJO_CHECK(warps_for_block(33) == 2);
  DOJO_CHECK(warps_for_block(256) == 8);
}

DOJO_TEST_MAIN()
