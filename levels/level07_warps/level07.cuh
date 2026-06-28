#pragma once

#include <cstddef>

// ---------------------------------------------------------------------------
// Level 7 — Warp programming.
//
// The one idea: 32 threads can talk without memory. A warp is the real
// execution unit — 32 lanes advancing as a single SIMD instruction. Because
// they move in lockstep, after any instruction their registers are mutually
// visible with NO barrier. Warp intrinsics let one lane read another lane's
// register directly, so register exchange is "free" relative to a shared-memory
// round trip.
//
// The new primitives (always the _sync variants, always with an explicit mask):
//   __shfl_down_sync(mask, v, d)  read v from lane (id + d)   — reduction workhorse
//   __shfl_sync(mask, v, srcLane) read v from a specific lane — broadcast
//   __ballot_sync(mask, pred)     32-bit mask of lanes where pred is true — vote
//
// The perf rule: a full-warp sum is five shuffles, no __shared__, no
// __syncthreads(). The production reduction stacks two levels —
//   warp-reduce within each warp  -> one partial per warp to __shared__
//   -> warp-reduce those partials
// — so a 256-thread block does ONE __syncthreads() instead of the five that
// Level 1's shared-memory tree needed. This is how CUB/Thrust reduce.
//
// The mask trap: a shuffle from an INACTIVE lane returns garbage. With a full
// warp use 0xffffffff; when a warp can diverge (the tail block, n not a multiple
// of 32) you must reduce only over active lanes — either mask out dead lanes'
// contributions (feed them the identity, 0 for a sum) or compute the live mask.
//
// Cliffhanger back to Level 6: the block reduction's tail still walked the
// shared tree under __syncthreads(); here the last warp does it barrier-free.
// ---------------------------------------------------------------------------

namespace dojo::level07 {

// Sum of in[0..n) computed on the GPU with a warp-shuffle reduction (warp ->
// shared partials -> warp). COMPLETE worked example — the warp-reduce pattern.
float reduce_sum_warp(const float *in, int n);

// Histogram of n bytes into num_bins (a byte value v increments bin
// v * num_bins / 256). Uses warp-aggregated atomics: lanes that target the same
// bin vote with __ballot_sync, and one lane does a single atomicAdd for the
// whole group — 32 atomics collapse toward 1. STUB — your job.
void histogram_warp(const unsigned char *in, int n, unsigned int *bins,
                    int num_bins);

} // namespace dojo::level07
