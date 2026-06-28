# Level 3 — The Memory Hierarchy

> **The question:** *Two kernels do the identical arithmetic. One is 10× faster.
> The only difference is where the data sat when the math units asked for it. Why
> is "where" worth 10×?*

Most beginners stop at Level 2 — they can write correct kernels. Production
engineers live *here*, because on a memory-bound machine (which a GPU is), the
memory hierarchy *is* the performance.

## The latency cliff

You already met this at Level 0; now make it concrete on the **RTX PRO 6000
Blackwell**. Each tier you fall through costs roughly an order of magnitude more
cycles. Drag the *reuse factor* to see why staging data in shared memory is the
whole game:

<div data-dojo="mem-hierarchy"></div>

| Space | Latency | Capacity (RTX PRO 6000) | Scope | You control it via |
|-------|---------|--------------------------|-------|--------------------|
| **Registers** | ~1 cyc | 256 KB / SM | per-thread | local variables |
| **Shared memory** | ~25 cyc | ≤ 128 KB / SM | per-block | `__shared__` (Level 5) |
| **L1 / read-only cache** | ~30 cyc | shares the 128 KB / SM | per-SM | `__ldg` / `const __restrict__` |
| **L2 cache** | ~250 cyc | on-chip, GPU-wide | whole GPU | (automatic) |
| **Global (GDDR7)** | ~600 cyc | 96 GB @ 1792 GB/s | whole GPU | `cudaMalloc` |
| **Constant** | ~register (if uniform) | 64 KB | read-only, broadcast | `__constant__` |

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

From here on, start reading memory access as hardware transactions, not scalar
loads. A warp issues 32 lane requests; the memory system groups them into one or
more transactions depending on alignment and contiguity. That transaction count is
why a correct strided kernel can still be slow.

!!! note "Constant and read-only memory: the cheap wins"
    - `__constant__` (64 KB) is perfect for small read-only data that *every*
      thread reads the same way — filter coefficients, transform matrices. A warp
      reading the same address gets a single broadcast.
    - Marking inputs `const T* __restrict__` lets the compiler route them through
      the read-only cache and assume no aliasing — often a free speedup.

## The keystone project: tiled transpose

At Level 2 you hit the wall: transpose can't be both read- and write-coalesced
with thread mapping alone. The fix is the archetype of all GPU optimization —
stage a tile in the scratchpad to convert a bad global access pattern into a good
one:

<svg class="dojo-diagram" viewBox="0 0 760 150" role="img" aria-label="Read a tile coalesced from global into shared, then write it out transposed but still coalesced.">
  <rect class="stroke-accent" x="20"  y="45" width="160" height="70" rx="8"/>
  <text class="mono" x="100" y="38" text-anchor="middle">global IN</text>
  <text class="mono fill-accent" x="100" y="86" text-anchor="middle">coalesced read</text>
  <rect class="stroke-faint"  x="300" y="45" width="160" height="70" rx="8"/>
  <text class="mono" x="380" y="38" text-anchor="middle">__shared__ tile</text>
  <text class="mono" x="380" y="80" text-anchor="middle">transpose</text>
  <text class="mono fill-accent" x="380" y="98" text-anchor="middle">happens here (~25 cyc)</text>
  <rect class="stroke-accent" x="580" y="45" width="160" height="70" rx="8"/>
  <text class="mono" x="660" y="38" text-anchor="middle">global OUT</text>
  <text class="mono fill-accent" x="660" y="86" text-anchor="middle">coalesced write</text>
  <g class="fill-accent">
    <path d="M180,80 l112,0 m-10,-6 l10,6 l-10,6 z"/>
    <path d="M460,80 l112,0 m-10,-6 l10,6 l-10,6 z"/>
  </g>
  <text class="mono" x="380" y="138" text-anchor="middle">both GDDR7 passes stay contiguous — the strided access is confined to fast shared memory</text>
</svg>

Step through it one row at a time:

<div data-dojo="tiled-transpose"></div>

You'll write this at Level 4/5. The two kernels differ by one staging buffer — and
that buffer is worth roughly 2× the bandwidth:

=== "Naive transpose (strided)"

    ```cpp { .annotate }
    __global__ void transpose_naive(const float* in, float* out, int w, int h) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x < w && y < h)
            out[x * h + y] = in[y * w + x];   // (1)!
    }
    ```

    1.  The read `in[y*w + x]` is coalesced (consecutive `x` → consecutive
        addresses). But the write `out[x*h + y]` makes consecutive lanes jump by
        `h` — **strided, uncoalesced.** You can't fix it by swapping x/y; that just
        moves the strided access to the read. One side is always bad.

=== "Tiled transpose (coalesced)"

    ```cpp { .annotate }
    __global__ void transpose_tiled(const float* in, float* out, int w, int h) {
        __shared__ float tile[32][33];                 // (1)!
        int x = blockIdx.x * 32 + threadIdx.x;
        int y = blockIdx.y * 32 + threadIdx.y;
        if (x < w && y < h)
            tile[threadIdx.y][threadIdx.x] = in[y*w + x];  // (2)!
        __syncthreads();                               // (3)!
        x = blockIdx.y * 32 + threadIdx.x;             // (4)!
        y = blockIdx.x * 32 + threadIdx.y;
        if (x < h && y < w)
            out[y*h + x] = tile[threadIdx.x][threadIdx.y];  // (5)!
    }
    ```

    1.  `[32][33]` — the `+1` padding column makes shared-memory accesses skip the
        32-way **bank conflict** the transpose would otherwise hit (Level 5).
    2.  Read coalesced into shared — consecutive lanes, consecutive global
        addresses.
    3.  Barrier: the whole block must finish loading before anyone reads the tile
        transposed. *Why here?* A lane reads a tile entry written by a **different**
        lane — without the barrier that's a race.
    4.  Recompute the output coordinates from the *swapped* block indices.
    5.  Write coalesced to global; the transpose came from reading the shared tile
        by column. The only strided access lives in ~25-cycle shared memory.

## Your reps

- **Tiled transpose** — first real use of `__shared__`; measure GB/s before and
  after (it roughly doubles).
- **Read-only convolution** — put the filter in `__constant__`, mark the image
  `__restrict__`, and watch the read-only cache do work for free.
- When you are ready for the hardware-level version of this model, read
  [Architecture Deep Dive](track-architecture.md).

??? question "Self-check"
    Vector add gets ~0 benefit from shared memory but tiled transpose gets ~2×.
    What single property explains the difference? *(Reuse factor. Vector add reads
    each byte once — nothing to amortize. Transpose's tiling doesn't add reuse,
    but it converts strided global writes into coalesced ones, recovering wasted
    bandwidth. Convolution/GEMM add genuine reuse on top.)*

→ Continue to [Level 4 — Performance Thinking](level04-performance.md)
