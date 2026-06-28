#pragma once

#include <cstddef>

// ---------------------------------------------------------------------------
// Level 5 — Shared-memory algorithms.
//
// The one idea: a block is a team with a shared scratchpad. Until now threads
// were loners. Here 256 threads cooperate through __shared__ memory (on-chip,
// ~25-cycle latency, a software-managed L1) and __syncthreads() (a block-wide
// barrier). The rhythm is always the same:
//
//     load a tile (plus a HALO of neighbors) global -> shared
//     __syncthreads()                                  (unconditional!)
//     compute, reading only the fast shared tile
//
// The perf rule: this turns Level 3/4's "hope the cache catches the reuse" into
// an explicit reuse factor. A radius-r box filter reads each input pixel up to
// (2r+1)^2 times. Stage it once in shared and every input crosses the slow DRAM
// boundary exactly once. Reuse factor = (output pixels * (2r+1)^2) / (tile
// fetches) is your speedup ceiling.
//
// Two traps the comments below force you to confront:
//   - __syncthreads() must be reached by EVERY thread or the block deadlocks.
//     Load behind a bounds guard, but sync OUTSIDE it. Never sync in a divergent
//     branch.
//   - Shared memory has 32 banks; a naive column access (tile[ty][tx] read as a
//     column) serializes 32-way. Padding the inner dimension (tile[N][N+1])
//     skews the layout — same conflict lesson as Level 4. Padding is the first
//     tool, not the only one: swizzle_demo.cu (a stretch) removes the same
//     conflict with XOR indexing (tile[row][col ^ row]) and no wasted memory —
//     the layout GEMM needs at Level 9.
//
// Cliffhanger: a tiled box filter still runs the same __syncthreads() three
// times in the block reduction's tail — but the last 32 threads are a single
// warp, already synchronized by physics. Level 6 deletes those barriers.
//
// Images: row-major single-channel float, pixel (x, y) at index y * width + x.
// Box filter = mean of the (2r+1)x(2r+1) neighborhood, edges clamped to the
// nearest valid pixel (so corners average a smaller-but-clamped window).
// ---------------------------------------------------------------------------

namespace dojo::level05 {

// (2r+1)x(2r+1) box blur (mean), edges clamped. Each block stages its output
// tile plus an r-wide halo in __shared__, syncs, then averages from shared.
// COMPLETE worked example — the load/sync/compute pattern this level teaches.
void box_filter_shared(const float *in, float *out, int w, int h, int radius);

// Same numeric result as box_filter_shared, but computed as two separable 1D
// passes (horizontal then vertical) so each output costs 2*(2r+1) reads instead
// of (2r+1)^2. STUB — your job.
void separable_blur(const float *in, float *out, int w, int h, int radius);

} // namespace dojo::level05
