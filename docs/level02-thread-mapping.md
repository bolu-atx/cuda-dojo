# Level 2 — Thread Mapping

> **The question:** *An image is a 2D grid of pixels. Your data is. Your threads
> can be too. How do you lay 16 million threads over a 4096×4096 image so the
> mapping is both correct and fast?*

This is where CUDA stops being "parallel for" and starts being design. The chain
you're building is:

```
pixel → thread → warp → block → grid
```

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
formula — once per axis:

```cpp
int x = blockIdx.x * blockDim.x + threadIdx.x;  // column
int y = blockIdx.y * blockDim.y + threadIdx.y;  // row
if (x >= width || y >= height) return;          // boundary guard
int idx = y * width + x;                         // row-major flatten
```

The indexing is still just `block * dim + thread` — now applied per dimension.
Convince yourself on the 1D widget; the 2D case is two of these:

<div data-dojo="thread-index"></div>

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

## Your reps (these all live or extend in the repo's `levels/`)

| Project | What it teaches |
|---------|-----------------|
| **image invert / threshold / brightness** | the 2D launch + boundary guard, trivial math |
| **pixel histogram** | a first scatter (sets up Level 7 atomics) |
| **crop** | output index ≠ input index; offset arithmetic |
| **transpose** | output `(x,y)` reads input `(y,x)` — the canonical coalescing trap |
| **rotation / resize** | non-integer source coords → gather + interpolation |

**Transpose is the keystone.** A naive transpose reads coalesced but *writes*
strided (or vice versa). You cannot win with thread mapping alone — which is
exactly the cliffhanger that forces you into the [memory
hierarchy](level03-memory-hierarchy.md) and shared-memory tiling at Level 4.

??? question "Self-check"
    For a 1920×1080 image with `block(32, 8)`, how many blocks does
    `dim3 grid(ceil_div(1920,32), ceil_div(1080,8))` create, and why is
    `block.x = 32` a deliberate choice? *(60×135 = 8100 blocks; 32 = warp size,
    so each warp's 32 lanes span one contiguous run of 32 pixels → coalesced.)*

→ Continue to [Level 3 — Memory Hierarchy](level03-memory-hierarchy.md)
