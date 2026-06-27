# Level 3 — The Memory Hierarchy

> **The question:** *Two kernels do the identical arithmetic. One is 10× faster.
> The only difference is where the data sat when the math units asked for it. Why
> is "where" worth 10×?*

Most beginners stop at Level 2 — they can write correct kernels. Production
engineers live *here*, because on a memory-bound machine (which a GPU is), the
memory hierarchy *is* the performance.

## The latency cliff

You already met this at Level 0; now make it concrete. Each tier you fall through
costs roughly an order of magnitude more cycles:

<div data-dojo="mem-latency"></div>

| Space | Latency | Scope | You control it via |
|-------|---------|-------|--------------------|
| **Registers** | ~1 cyc | per-thread | local variables |
| **Shared memory** | ~20 cyc | per-block | `__shared__` (Level 5) |
| **L1 / read-only cache** | ~30 cyc | per-SM | `__ldg` / `const __restrict__` |
| **L2 cache** | ~200 cyc | whole GPU | (automatic) |
| **Global (DRAM)** | ~400–800 cyc | whole GPU | `cudaMalloc` |
| **Constant** | ~register (if uniform) | read-only, broadcast | `__constant__` |

## The two ways to beat latency

You can't make the trip to DRAM faster. You have exactly two weapons:

1. **Hide it** — have so many resident warps that while warp A waits 500 cycles,
   warps B…Z keep the ALUs busy. This is *latency hiding* and it's free if you have
   enough parallelism (occupancy). It's the GPU's default coping mechanism.

2. **Avoid it** — move data *up* the hierarchy and *reuse* it. If a value is read
   10 times, pay the 500-cycle DRAM trip once, stash it in shared memory (~20 cyc),
   and serve the other 9 reads cheaply. This is *data reuse*, and it's the entire
   premise of tiling.

The key ratio is the **reuse factor**: how many times does each byte you fetch
from DRAM get used? Vector add reuses each byte *once* — nothing to cache, pure
bandwidth. A convolution or matrix multiply reuses each input many times — huge
opportunity to stage it in shared memory.

!!! note "Constant and read-only memory: the cheap wins"
    - `__constant__` (64 KB) is perfect for small read-only data that *every*
      thread reads the same way — filter coefficients, transform matrices. A warp
      reading the same address gets a single broadcast.
    - Marking inputs `const T* __restrict__` lets the compiler route them through
      the read-only cache and assume no aliasing — often a free speedup.

## The keystone project: tiled transpose

At Level 2 you hit the wall: transpose can't be both read- and write-coalesced
with thread mapping alone. The fix is the archetype of all GPU optimization:

```
read a TILE from global memory, coalesced  →  stash it in __shared__  →
write it out transposed, also coalesced
```

The strided access is confined to fast shared memory; both global-memory passes
stay coalesced. You'll write this at Level 4/5, but the *idea* — "stage a tile in
the scratchpad to convert a bad global access pattern into a good one" — is Level
3's whole payload.

## Your reps

- **Tiled transpose** — first real use of `__shared__`; measure GB/s before and
  after (it roughly doubles).
- **Read-only convolution** — put the filter in `__constant__`, mark the image
  `__restrict__`, and watch the read-only cache do work for free.

??? question "Self-check"
    Vector add gets ~0 benefit from shared memory but tiled transpose gets ~2×.
    What single property explains the difference? *(Reuse factor. Vector add reads
    each byte once — nothing to amortize. Transpose's tiling doesn't add reuse,
    but it converts strided global writes into coalesced ones, recovering wasted
    bandwidth. Convolution/GEMM add genuine reuse on top.)*

→ Continue to [Level 4 — Performance Thinking](level04-performance.md)
