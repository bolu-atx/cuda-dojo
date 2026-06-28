#pragma once

#include <cstddef>

// ---------------------------------------------------------------------------
// Level 4 — Performance thinking.
//
// The one idea: every kernel is limited by one of exactly two walls — memory
// bandwidth or compute throughput. Which wall is set by the kernel's *arithmetic
// intensity* (FLOPs per byte moved), and you should compute that number BEFORE
// you optimize anything.
//
//   Left of the roofline ridge  -> memory-bound. Math units starve. Fixing math
//                                  is pointless; fix coalescing and reuse.
//   Right of the ridge          -> compute-bound. Now fewer/faster instructions
//                                  (and tensor cores) pay off.
//
// A 3x3 blur reads ~9 floats and does ~9 FLOPs per output => AI < 1 FLOP/byte =>
// deeply memory-bound. So is Sobel. The job at this level is to *measure* (the
// demo reports achieved GB/s and % of peak) and to name which of the four memory
// sins — uncoalesced access, bank conflicts, divergence, low occupancy — is
// holding you back.
//
// Images: row-major single-channel float, pixel (x, y) at index y * width + x.
// ---------------------------------------------------------------------------

namespace dojo::level04 {

// Box blur of radius r: mean over the (2r+1)x(2r+1) window, edges clamped.
// Baseline — straightforward, memory-bound.
void box_blur_naive(const float *in, float *out, int w, int h, int radius);

// Same result; your optimized version (improve coalescing / exploit reuse).
void box_blur_opt(const float *in, float *out, int w, int h, int radius);

// Sobel gradient magnitude: sqrt(Gx^2 + Gy^2) over the 3x3 neighborhood.
// A second memory-bound stencil to compare arithmetic intensity against.
void sobel(const float *in, float *out, int w, int h);

} // namespace dojo::level04
