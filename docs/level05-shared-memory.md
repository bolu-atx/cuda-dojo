# Level 5 — Shared-Memory Algorithms

> **The question:** *A 3×3 filter reads every pixel up to 9 times. From DRAM, that's
> 9 trips of ~500 cycles each. How do you pay for each pixel exactly once?*

Until now threads were loners. At this level a **block becomes a team**: 256
threads with a shared scratchpad (`__shared__`) and a barrier (`__syncthreads()`).
This is where you implement the "avoid latency by reusing data" idea from Level 3.

## The two new primitives

```cpp
__shared__ float tile[18][18];   // per-BLOCK scratchpad, ~20-cycle latency
// ... every thread loads its piece ...
__syncthreads();                 // barrier: nobody proceeds until ALL arrive
// ... now everyone can safely read what everyone else loaded ...
```

- `__shared__` memory is allocated **per block**, lives on-chip, and is visible to
  every thread in the block. Think of it as a software-managed L1.
- `__syncthreads()` is a block-wide barrier. It's how you make cooperation
  *correct*: load phase → **barrier** → compute phase. Skip the barrier and you
  read garbage your teammates haven't written yet.

!!! danger "`__syncthreads()` must be hit by every thread or you deadlock"
    Never put `__syncthreads()` inside a divergent branch (`if (x < width)`) where
    some threads skip it. Some lanes wait forever for lanes that already returned.
    Load with the guard, then sync *unconditionally*.

## The pattern: load a tile (with halo), sync, compute

A stencil/convolution needs neighbors. So each block loads its output tile **plus
a halo** of surrounding pixels into shared memory, synchronizes, then every thread
computes its output reading only fast shared memory:

```
┌─────── halo (extra border pixels) ───────┐
│  · · · · · · · · · · · · · · · · · · ·    │
│  · ┌───────────────────────────────┐ ·   │
│  · │   the block's output tile      │ ·   │   load this whole thing once
│  · │   (one thread per pixel)       │ ·   │   from global → shared,
│  · └───────────────────────────────┘ ·   │   then compute from shared
│  · · · · · · · · · · · · · · · · · · ·    │
└───────────────────────────────────────────┘
```

Each input pixel crosses the slow DRAM boundary **once** even though it's read by
up to 9 different output computations. That's the whole win — the reuse factor you
identified at Level 3, realized in code.

## Cooperative algorithms beyond stencils

Shared memory also powers algorithms where the block computes a *collective*
result. You already saw the shape at Level 1 — the in-block tree reduction lives
here:

<div data-dojo="reduction"></div>

The same load→sync→combine→sync rhythm builds **prefix scan** (running totals),
**local histograms** (privatize bins in shared memory, then merge — Level 7), and
the **integral image**. Master the rhythm once and these all fall out.

!!! tip "Don't forget bank conflicts (Level 4)"
    Shared memory is fast *only* when the 32 lanes of a warp hit 32 different
    banks. A naive `tile[ty][tx]` column access conflicts; pad the inner dimension
    (`tile[TILE][TILE+1]`) to skew the layout. Verify with `ncu`'s
    "shared memory bank conflicts" metric, don't assume.

## Your reps

| Project | New skill |
|---------|-----------|
| **box filter** (worked) | tile + halo loading, the load/sync/compute rhythm |
| **separable blur** (your turn) | two passes (row, col) — half the work |

Stretch reps once those land: revisit the **tiled transpose** with shared-memory
padding, a per-block **local histogram** (privatized bins → bridge to Level 7),
and a **block prefix scan** (up-sweep / down-sweep over shared memory).

??? question "Self-check"
    Your tiled box filter loads a 16×16 output tile with a 1-pixel halo. How big
    is the shared tile, and how many input pixels does each block fetch from
    global memory vs. how many output pixels it produces? *(18×18 = 324 fetched;
    256 produced. The 324 fetches replace up to 256×9 = 2304 naive global reads —
    that ratio is your speedup ceiling.)*

→ Continue to [Level 6 — Warp Programming](level06-warps.md)
