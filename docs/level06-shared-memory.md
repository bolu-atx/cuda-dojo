# Level 6 — Shared-Memory Algorithms

> **The question:** *A 3×3 filter reads every pixel up to 9 times. From DRAM, that's
> 9 trips of ~500 cycles each. How do you pay for each pixel exactly once?*

Until now threads were loners. At this level a **block becomes a team**: 256
threads with a shared scratchpad (`__shared__`) and a barrier (`__syncthreads()`).
This is where you implement the "avoid latency by reusing data" idea from Level 4.

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
identified at Level 4, realized in code.

## Cooperative algorithms beyond stencils

Shared memory also powers algorithms where the block computes a *collective*
result. You already saw the shape at Level 1 — the in-block tree reduction lives
here:

<div data-dojo="reduction"></div>

The same load→sync→combine→sync rhythm builds **prefix scan** (running totals),
**local histograms** (privatize bins in shared memory, then merge — Level 9), and
the **integral image**. Master the rhythm once and these all fall out.

!!! tip "Don't forget bank conflicts (Level 5)"
    Shared memory is fast *only* when the 32 lanes of a warp hit 32 different
    banks. A naive `tile[ty][tx]` column access conflicts; pad the inner dimension
    (`tile[TILE][TILE+1]`) to skew the layout. Verify with `ncu`'s
    "shared memory bank conflicts" metric, don't assume.

## Padding vs. swizzling

Padding is the *first* tool for bank conflicts, not the only one. Shared memory has
32 banks; `bank(addr) = (addr / 4) % 32`. In a clean 32-wide tile, element
`(row, col)` lands in bank `col`. A transpose reads a whole **column** (`col`
fixed, `row` sweeps 0–31) — every lane hits the *same* bank `col`: a 32-way
conflict. Padding to `tile[32][33]` skews each row by one bank so a column spreads
across all 32. **Swizzling** gets the same result by permuting the index instead of
the layout — store the column at `col ^ row`:

```cpp
tile[row][col ^ row] = ...;   // store, then read back with the SAME xor
```

XOR is a bijection for each fixed operand, so it's conflict-free in *both*
directions, on a clean power-of-two square:

```
            bank hit by lane = row (col fixed) on the transpose read
  col=0 :  pad 0 1 2 .. 31   swizzle 0 1 2 .. 31   (both: all 32 distinct ✓)
  naive 0 :          0 0 0 .. 0   ← every lane → bank 0, 32-way conflict ✗
```

So why ever swizzle if padding already works? Two costs padding pays that swizzle
avoids: it **wastes shared memory** (the scarce resource that caps occupancy), and
its non-power-of-two stride **breaks 128-bit vectorized `ld.shared`** (a stride of
33 floats is unaligned for `float4`). On a transpose neither bites — but predict
the bank conflicts per warp for the naive, padded, and swizzled kernels, then prove
it:

```bash
ncu --set full ./level06_swizzle_demo   # metric: Shared Memory Bank Conflicts
```

Naive spikes; padded and swizzled are both ~0. On a plain transpose padding is
fine, so swizzle looks optional — but at [Level 11](level11-multi-kernel.md), a
tiled GEMM can't spare the shared memory or give up vectorized loads, so swizzle
stops being a choice.

## Your reps

| Project | New skill |
|---------|-----------|
| **box filter** (worked) | tile + halo loading, the load/sync/compute rhythm |
| **separable blur** (your turn) | two passes (row, col) — half the work |

Stretch reps once those land: revisit the **tiled transpose** with shared-memory
padding, **shared-memory swizzling** (XOR indexing — conflict-free with no padding;
run `level06_swizzle_demo`), a per-block **local histogram** (privatized bins →
bridge to Level 9), and a **block prefix scan** (up-sweep / down-sweep over shared
memory).

??? question "Self-check"
    Your tiled box filter loads a 16×16 output tile with a 1-pixel halo. How big
    is the shared tile, and how many input pixels does each block fetch from
    global memory vs. how many output pixels it produces? *(18×18 = 324 fetched;
    256 produced. The 324 fetches replace up to 256×9 = 2304 naive global reads —
    that ratio is your speedup ceiling.)*

??? question "Self-check — swizzle"
    A 32×32 tile, transpose read (column access). What bank does lane `row` hit
    with a naive `tile[row][col]`, a padded `tile[row][col]` over `[32][33]`, and a
    swizzled `tile[row][col ^ row]`? *(Naive: bank `col` for every lane → 32-way
    conflict. Padded: each row shifts by one bank, so lanes hit `(col + row) % 32`
    → all distinct. Swizzle: lanes hit `col ^ row` → all distinct. Padding burns a
    column of smem to do it; swizzle keeps the clean power-of-two stride.)*

You made the *block* cooperate through shared memory and a barrier. But the last
32 threads of any block are a single warp — and they can trade values through
registers, with no `__shared__` and no `__syncthreads()` at all. That tighter,
faster cooperation unit is next.

→ Continue to [Level 7 — Warp Programming](level07-warps.md)
