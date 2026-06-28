#pragma once

#include <cstddef>

// ---------------------------------------------------------------------------
// Level 4 — Memory hierarchy.
//
// The one idea: where data lives dominates speed. Each tier you fall through —
// registers (~1 cyc) -> shared (~25) -> L1 -> L2 (~250) -> global (~600) — costs
// roughly an order of magnitude more cycles. There are exactly two ways to beat
// latency: hide it (enough resident warps) or avoid it (move data up the
// hierarchy and *reuse* it).
//
// Two problems, each as a complete baseline + an optimized version you write:
//
//   transpose:  naive  (one end strided, the Level 2 cliffhanger)   [baseline]
//               tiled  (stage a tile in __shared__, both ends coalesced) [stub]
//
//   3x3 blur:   global   (filter in registers, inputs straight from global) [baseline]
//               constant (filter in __constant__, inputs via read-only path) [stub]
//
// Reuse factor is the lens: transpose reuses each byte once but still wins ~2x
// from tiling because shared memory converts a strided access into a coalesced
// one. A 3x3 blur reuses each input pixel up to 9 times across neighboring
// outputs — that reuse is what the read-only/constant path and (later) shared
// tiling exploit.
//
// Images: row-major single-channel float, pixel (x, y) at index y * width + x.
// ---------------------------------------------------------------------------

namespace dojo::level04 {

// out[x*h + y] = in[y*w + x]. Baseline: coalesced read, strided write.
void transpose_naive(const float *in, float *out, int w, int h);

// Same result via a shared-memory tile so both global ends are coalesced.
void transpose_tiled(const float *in, float *out, int w, int h);

// 3x3 box blur (mean of the 3x3 neighborhood), edges clamped. Baseline.
void blur3x3_global(const float *in, float *out, int w, int h);

// Same 3x3 blur with the filter in __constant__ and inputs through the
// read-only cache (const __restrict__).
void blur3x3_constant(const float *in, float *out, int w, int h);

} // namespace dojo::level04
