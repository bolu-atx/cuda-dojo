#pragma once

#include <cstddef>

// ---------------------------------------------------------------------------
// Level 2 — Thread mapping.
//
// The one idea: you design the thread -> data mapping. Images are a 2D grid of
// pixels, so we launch a 2D grid of threads (dim3) and choose how each level of
// the thread hierarchy lands on the data.
//
// The rule that governs performance: threadIdx.x is the fastest-varying axis, so
// a warp's 32 lanes differ in x. For coalesced global access, x must map to the
// contiguous direction of row-major memory — the *column*. Get it backwards
// (index as x*height + y, or map x down a column) and every load fans out into
// many transactions, costing ~10-30x bandwidth.
//
// Memory layout for all images here: row-major, single-channel float ("grayscale"),
// pixel (x, y) lives at index y * width + x.
// ---------------------------------------------------------------------------

namespace dojo::level02 {

// out[y*w+x] = maxval - in[y*w+x]. The canonical 2D launch + boundary guard.
void invert(const float *in, float *out, int w, int h, float maxval);

// Copy an outW x outH window whose top-left corner is (x0, y0) out of the
// inW x inH source. Teaches: output index != input index (offset arithmetic).
void crop(const float *in, int inW, int inH, float *out, int x0, int y0,
          int outW, int outH);

// out is the h x w transpose of the w x h input: out[x*h + y] = in[y*w + x].
// The keystone coalescing trap — see transpose_naive.cu.
void transpose(const float *in, float *out, int w, int h);

} // namespace dojo::level02
