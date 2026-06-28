# Level 11 — Multi-Kernel Algorithms

> **The question:** *Your pipeline is five kernels in a row. Between each, the
> intermediate result is written to DRAM and read straight back. What if the bytes
> never had to leave the chip?*

Real programs aren't one kernel — they're a *graph* of them. This level is about
the costs that only appear when kernels compose: launch overhead, the global
synchronization between stages, and the DRAM round-trips for intermediates.

## Why not just one giant kernel?

Because **blocks can't synchronize with each other with ordinary block barriers** —
the cross-block rung of Level 8's [scope ladder](level08-synchronization-scopes.md).
Any algorithm with a global barrier — "every block must finish step A before *any*
block starts step B" — needs that barrier to be a **kernel boundary**. The kernel
launch *is* the global sync. So multi-stage algorithms are naturally multi-kernel:

```
histogram kernel  →  scan kernel  →  scatter kernel        (a radix sort pass)
reduce-max kernel →  normalize kernel →  threshold kernel   (an imaging stage)
```

Each arrow is a global synchronization point you got for free by ending one kernel
and starting the next.

## The three costs of composition

### 1. Launch overhead
Each launch costs a few microseconds of CPU→GPU dispatch. Negligible for a 5 ms
kernel; *dominant* for a 3 µs kernel called 10,000 times. Fixes: do more work per
launch, batch, or amortize with **CUDA Graphs** (orchestration patterns, Level 13/14) which record a whole
kernel sequence and replay it with one launch.

### 2. Intermediate DRAM traffic
Stage A writes 100 MB to global memory; stage B immediately reads it back. That's
200 MB of bandwidth spent on data that never needed to persist. On a memory-bound
machine this is often the single biggest waste in a pipeline.

### 3. Synchronization stalls
If stage B waits on *all* of stage A, you can't overlap them. Sometimes you can
restructure into independent sub-streams (Level 10) so parts overlap.

## Kernel fusion: the big lever

**Fusion** merges adjacent kernels so intermediates stay in registers or shared
memory instead of round-tripping through DRAM:

```
// before: 3 kernels, 2 DRAM round-trips for tmp1, tmp2
scale<<<>>>(in, tmp1);  bias<<<>>>(tmp1, tmp2);  relu<<<>>>(tmp2, out);

// after: 1 kernel, intermediates live in registers
fused<<<>>>(in, out);   // out = relu(scale(in) + bias)
```

For memory-bound elementwise chains this is close to an N× win (N = number of
fused passes), because you read the input once and write the output once instead
of N times each. This is exactly why deep-learning compilers fuse so aggressively.

!!! tip "When *not* to fuse"
    Fusion raises register/shared-memory pressure, which can lower occupancy
    (Level 5) and *cost* you. And it hurts modularity. Fuse the memory-bound
    elementwise chains; keep compute-heavy stages (a cuBLAS GEMM) separate. As
    always: measure with `ncu`, don't assume.

## Streams still apply across kernels

Independent stages or independent data can overlap across streams just like
transfers did at Level 10:

<div data-dojo="streams"></div>

## Your reps

- **GEMM** — the canonical multi-stage tiled algorithm: load tiles → multiply →
  accumulate, looping over the K dimension. Register-blocked GEMM is the rite of
  passage; then compare to cuBLAS (Level 9).

    !!! tip "Here padding stops working — swizzle instead"
        GEMM is where the [Level 6](level06-shared-memory.md#padding-vs-swizzling)
        swizzle stretch pays off. Its shared tiles fight for the smem that caps
        occupancy, and the inner loop relies on 128-bit vectorized `ld.shared`. A
        `[N][N+1]` pad wastes that smem *and* breaks the power-of-two stride
        `float4` needs — so a fast GEMM removes bank conflicts with **XOR
        swizzling** (`tile[row][col ^ row]`), not padding.
- **Fuse an imaging chain** — take a `scale → bias → clamp` sequence on an image,
  measure the 3-kernel version, then fuse into one and measure again. Report the
  DRAM bytes saved and the speedup.

??? question "Self-check"
    A 4-stage elementwise filter on a 200 MB image runs as 4 separate kernels.
    Roughly how much DRAM traffic does fusing them into one kernel save, and why
    is the speedup large? *(Unfused: each stage reads+writes ~200 MB → ~1.6 GB.
    Fused: read once + write once → ~400 MB. ~4× less traffic, and since the chain
    is memory-bound, ~4× faster.)*

→ Continue to [Level 12 — Production Optimization](level12-optimization.md)
