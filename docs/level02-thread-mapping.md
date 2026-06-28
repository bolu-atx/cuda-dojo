# Level 2 — Thread Mapping

> **The question:** *An image is a 2D grid of pixels. Your data is. Your threads
> can be too. How do you lay 16 million threads over a 4096×4096 image so the
> mapping is both correct and fast?*

This is where CUDA stops being "parallel for" and starts being design. The chain
you're building maps your *data* onto the hardware *hierarchy*:

<svg class="dojo-diagram" viewBox="0 0 760 110" role="img" aria-label="pixel maps to thread maps to warp maps to block maps to grid">
  <g class="mono">
    <rect class="fill-accent" x="10"  y="38" width="110" height="34" rx="5"/><text x="65"  y="60" text-anchor="middle" fill="#0b1500">pixel</text>
    <rect class="stroke-faint" x="170" y="38" width="110" height="34" rx="5"/><text x="225" y="60" text-anchor="middle">thread</text>
    <rect class="stroke-faint" x="330" y="38" width="110" height="34" rx="5"/><text x="385" y="60" text-anchor="middle">warp (32)</text>
    <rect class="stroke-faint" x="490" y="38" width="110" height="34" rx="5"/><text x="545" y="60" text-anchor="middle">block</text>
    <rect class="stroke-faint" x="650" y="38" width="100" height="34" rx="5"/><text x="700" y="60" text-anchor="middle">grid</text>
  </g>
  <g class="fill-accent">
    <path d="M120,55 l46,0 m-8,-5 l8,5 l-8,5 z"/>
    <path d="M280,55 l46,0 m-8,-5 l8,5 l-8,5 z"/>
    <path d="M440,55 l46,0 m-8,-5 l8,5 l-8,5 z"/>
    <path d="M600,55 l46,0 m-8,-5 l8,5 l-8,5 z"/>
  </g>
  <text class="mono" x="20" y="24">your data</text>
  <text class="mono" x="650" y="24">the hardware</text>
  <text class="mono" x="385" y="92" text-anchor="middle">32 lanes in lockstep — the unit that coalesces</text>
</svg>

You choose how each level of that hierarchy lands on your data. Get it right and
the hardware rewards you; get it backwards and you'll do the same math 10× slower.

## 2D launches

`dim3` lets blocks and grids be multi-dimensional. For an image:

```cpp
dim3 block(16, 16);                       // 256 threads, a 16×16 tile
dim3 grid(ceil_div(width, 16),
          ceil_div(height, 16));
kernel<<<grid, block>>>(img, width, height);
```

Inside the kernel each thread finds *its* pixel with the 2D version of the Level 1
formula — once per axis. The mapping you pick decides whether a warp's 32 lanes
land on **contiguous** memory (fast) or **scattered** memory (slow):

=== "Correct: x → column"

    ```cpp { .annotate }
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // (1)!
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // (2)!
    if (x >= width || y >= height) return;          // (3)!
    int idx = y * width + x;                         // (4)!
    out[idx] = f(in[idx]);
    ```

    1.  `threadIdx.x` is the **fastest-varying** axis, so map it to the **column**.
    2.  `y` (the slow axis) maps to the row.
    3.  Two-sided boundary guard — the image is rarely a multiple of the block.
    4.  Row-major flatten. Because consecutive lanes differ in `x`, they read
        `idx, idx+1, idx+2 …` — **one coalesced transaction**.

=== "Wrong: x → row"

    ```cpp { .annotate }
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // (1)!
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = x * height + y;                         // (2)!
    out[idx] = f(in[idx]);
    ```

    1.  Same index math…
    2.  …but flattening as `x * height + y` makes consecutive lanes (differing in
        `x`) jump by `height` floats. The warp now touches 32 *different* 128-byte
        segments — **strided, uncoalesced, ~1/32 of peak bandwidth.** The kernel is
        still *correct*; it's just slow for a reason no test will show you.

Drop a `dim3 block` over an image below and **click a pixel**. Watch the warp's 32
addresses in the linear-memory strip, and flip the mapping toggle to feel the
difference:

<div data-dojo="thread-map-2d"></div>

!!! danger "The mistake everyone makes once: x/y swapped"
    `threadIdx.x` is the **fastest-varying** axis. Consecutive threads (and thus a
    warp's 32 lanes) differ in `x`. So `x` must map to the **column** (the
    contiguous direction in row-major memory). Map `x` to rows instead and your
    warp reads strided addresses — uncoalesced, and you've thrown away most of
    your bandwidth before writing a single optimization. Why that matters is the
    next widget.

## Why the mapping *is* a performance decision

Memory isn't 2D. `img[y*width + x]` is one long row-major array. A warp is 32
threads with consecutive `threadIdx.x`, so a good mapping makes those 32 threads
touch 32 *consecutive* floats — which the hardware serves in one transaction.
This is **coalescing**, and it's the single biggest reason a "correct" kernel can
still be slow. Here's a warp's access pattern as you vary the stride between
threads:

<div data-dojo="coalescing"></div>

Stride 1 (threads map to consecutive columns) ⇒ ~full efficiency. Any larger
stride (threads map down a column, or you index `x*height + y`) wastes bus
bandwidth. We'll make this quantitative at [Level 4](level04-performance.md);
for now just **always make `threadIdx.x` run along contiguous memory.**

## The wider menu

| Project | What it teaches |
|---------|-----------------|
| **image invert / threshold / brightness** | the 2D launch + boundary guard, trivial math |
| **pixel histogram** | a first scatter (sets up Level 7 atomics) |
| **crop** | output index ≠ input index; offset arithmetic |
| **transpose** | output `(x,y)` reads input `(y,x)` — the canonical coalescing trap |
| **rotation / resize** | non-integer source coords → gather + interpolation |

## The exercises — `levels/level02_thread_mapping/`

Three of those reps are wired up for you. **invert** is a complete worked
example: read it first, it's the pattern every other kernel here copies. The
other two are stubs — the host plumbing is written, the kernel body is a `TODO`,
and the tests are the spec. Fill them in until `make level02-test` is green.

| File | Status | The payoff |
|------|--------|------------|
| `invert.cu` | ✅ **worked example** | See the canonical 2D launch: `block(32,8)`, two-sided guard, `x`→column so a warp's 32 lanes read one contiguous run. |
| `crop.cu` | 📝 **your turn** | The output pixel `(x,y)` is **not** the input pixel: it reads `(x+x0, y+y0)`. Get fluent with "which thread owns which element," and notice the input/output allocations are now different sizes. |
| `transpose_naive.cu` | 📝 **your turn**, the keystone | `out[x*h+y] = in[y*w+x]`. Implement it, then **predict before you profile**: one end is coalesced, the other is strided by `h`. Roughly what fraction of peak bandwidth survives? |

**Transpose is the keystone.** A naive transpose reads coalesced but *writes*
strided (or vice versa). You cannot win with thread mapping alone — which is
exactly the cliffhanger that forces you into the [memory
hierarchy](level03-memory-hierarchy.md) and shared-memory tiling at Level 3.

!!! tip "Prove it, don't trust it"
    `make level02-test` runs each kernel against a CPU reference at awkward sizes
    (1023×577, 1×1, single-block) so an off-by-one in the boundary guard or a
    swapped `(w,h)` actually fails. Then run `level02_demo` and compare the
    `invert` GB/s against the `transpose` GB/s on the same image — same bytes
    moved, very different bandwidth. That gap *is* the lesson.

??? question "Self-check"
    For a 1920×1080 image with `block(32, 8)`, how many blocks does
    `dim3 grid(ceil_div(1920,32), ceil_div(1080,8))` create, and why is
    `block.x = 32` a deliberate choice? *(60×135 = 8100 blocks; 32 = warp size,
    so each warp's 32 lanes span one contiguous run of 32 pixels → coalesced.)*

→ Continue to [Level 3 — Memory Hierarchy](level03-memory-hierarchy.md)
