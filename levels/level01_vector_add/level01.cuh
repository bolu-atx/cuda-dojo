#pragma once

#include <cstddef>

// ---------------------------------------------------------------------------
// Level 1 — CUDA programming basics.
//
// Each function below is the canonical "full flow" for one problem: allocate
// device memory, copy host->device, launch the kernel, copy device->host, free.
// Keeping the whole flow in one place makes the pattern obvious and lets the
// demo and the tests share exactly the same code path.
//
// Inputs/outputs are plain host pointers so callers don't need to touch CUDA.
// ---------------------------------------------------------------------------

namespace dojo::level01 {

// out[i] = a[i] + b[i]
void vector_add(const float *h_a, const float *h_b, float *h_out, int n);

// out[i] = alpha * x[i] + y[i]   (single-precision a*X plus Y)
void saxpy(float alpha, const float *h_x, const float *h_y, float *h_out, int n);

// returns sum(in[0..n)) computed on the GPU via a two-pass block reduction.
float reduce_sum(const float *h_in, int n);

} // namespace dojo::level01
