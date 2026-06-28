# Level 6 — Warp Programming

> **The question:** *Your block reduction uses shared memory and three
> `__syncthreads()`. But the last 32 threads are a single warp running in
> lockstep — they're already synchronized by physics. Can they swap data without
> touching memory at all?*

Yes. This level is where you stop pretending threads are independent and exploit
the fact that 32 of them *are the same SIMD instruction*. Warp intrinsics let
lanes read each other's registers directly — no shared memory, no barrier.

## The warp is the real execution unit

Recall the SIMT widget — 32 lanes, one instruction stream:

<div data-dojo="simt"></div>

Because lanes advance together, register exchange between them is "free" relative
to a shared-memory round trip. The modern intrinsics are **`_sync`-suffixed** and
take an active mask (use `0xffffffff` for a full warp):

| Intrinsic | What it does |
|-----------|--------------|
| `__shfl_sync(mask, val, srcLane)` | read `val` from another lane's register |
| `__shfl_down_sync(mask, val, d)` | read from lane `id + d` (the reduction workhorse) |
| `__shfl_xor_sync(mask, val, m)` | butterfly exchange — full reductions/scans |
| `__ballot_sync(mask, pred)` | 32-bit mask of which lanes have `pred` true |
| `__any_sync` / `__all_sync` | warp-wide OR / AND of a predicate |
| `__activemask()` | which lanes are currently active |

## Warp reduction: shared memory, deleted

A full-warp sum in five shuffles, no `__shared__`, no `__syncthreads()`:

```cpp
__device__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;   // lane 0 holds the warp's total
}
```

It's the same halving tree you stepped through at Level 1 — but the data moves
through the register file instead of shared memory:

<div data-dojo="reduction"></div>

The production reduction pattern stacks the two: **warp-reduce within each warp →
write one partial per warp to shared memory → warp-reduce those partials.** One
`__syncthreads()` total instead of five. This is how CUB and Thrust do it under
the hood.

!!! warning "Always use the `_sync` variants and the right mask"
    The old maskless `__shfl` is deprecated and unsafe on divergent warps. Pass an
    explicit mask. If your warp can diverge (e.g., the tail block), compute the
    mask with `__ballot_sync` / `__activemask` rather than assuming `0xffffffff` —
    a shuffle from an inactive lane returns garbage.

## Warp aggregation: cheap atomics

Hammering a global counter with 32 `atomicAdd`s per warp serializes them. Instead,
have the warp **vote and aggregate**: `__ballot_sync` counts how many lanes want to
increment, lane 0 does *one* `atomicAdd` of the total, then `__shfl_sync`
broadcasts the base offset back. 32 atomics → 1. This is the core trick behind
fast histograms and stream compaction.

## Your reps

| Project | Intrinsic focus |
|---------|-----------------|
| **warp reduction** (worked) | `__shfl_down_sync` — then benchmark vs the shared-mem version |
| **warp histogram** (your turn) | `__ballot_sync` + warp-aggregated `atomicAdd` |

Stretch rep: a **warp prefix sum** with `__shfl_up_sync` — an inclusive scan in
log₂32 = 5 steps.

??? question "Self-check"
    Why does a warp shuffle need no `__syncthreads()` but a block-wide reduction
    does? *(The 32 lanes of a warp execute in lockstep — they're literally one
    SIMD instruction, so after any instruction their registers are mutually
    visible with no barrier. Threads in *different* warps are scheduled
    independently, so crossing warp boundaries (via shared memory) requires an
    explicit barrier.)*

→ Continue to [Level 7 — CUDA Libraries](level07-libraries.md)
