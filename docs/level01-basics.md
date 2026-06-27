# Level 1 — CUDA Programming Basics

> **The question:** *You wrote one scalar function. How does it become 16 million
> parallel operations — and how does each copy know which element it owns?*

This level is the smallest complete CUDA program: move data to the GPU, run a
function across a grid of threads, move the result back. Everything else in the
dojo is a refinement of this loop.

## The three function-space qualifiers

CUDA adds a prefix that says *where code runs*:

| Qualifier | Runs on | Callable from |
|-----------|---------|---------------|
| `__global__` | GPU | host (this is a **kernel** — launched with `<<<>>>`) |
| `__device__` | GPU | GPU only (helper functions) |
| `__host__` | CPU | CPU (the default; combine `__host__ __device__` for both) |

A `__global__` function is the entry point. It always returns `void` — results
come back through memory you pass in.

## The launch and the thread hierarchy

```cpp
vector_add_kernel<<<grid, block>>>(d_a, d_b, d_out, n);
```

The `<<<grid, block>>>` says: launch `grid` **blocks**, each containing `block`
**threads**. Every thread runs the *same* kernel body but gets unique built-in
coordinates:

- `threadIdx.x` — my index within my block
- `blockIdx.x` — my block's index within the grid
- `blockDim.x` — threads per block
- `gridDim.x` — blocks in the grid

The single most important line in beginner CUDA computes a thread's **global
index** from these:

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

That's just flattening a 2-level hierarchy into one number. Play with it — find
the thread that owns element 19:

<div data-dojo="thread-index"></div>

!!! tip "Why blocks at all? Why not one flat array of threads?"
    A **block** is scheduled onto a single SM, and threads in a block can share a
    scratchpad and synchronize (Level 5). Blocks are the unit of cooperation;
    the grid is the unit of scale. Blocks can't (cheaply) talk to each other —
    that constraint is what makes GPUs scalable across SM counts.

## Guarding the boundary, and the grid-stride loop

`n` is rarely a multiple of `blockDim`, so you launch *at least* enough threads
and have extras do nothing:

```cpp
if (idx < n) out[idx] = a[idx] + b[idx];   // bounds guard
```

The idiom we actually use in the repo is the **grid-stride loop**, which works for
*any* launch size — each thread strides across the array by the total thread
count:

```cpp
for (int i = idx; i < n; i += blockDim.x * gridDim.x)
    out[i] = a[i] + b[i];
```

Move the "total threads" slider below it; the work stays balanced no matter how
many threads you launch:

<div data-dojo="grid-stride"></div>

## The memory dance

The GPU can't touch your `std::vector`. The canonical flow (this is exactly
`vector_add` in `levels/level01_vector_add/vector_add.cu`):

```cpp
cudaMalloc(&d_a, bytes);                                  // 1. allocate on device
cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);     // 2. copy in
vector_add_kernel<<<grid, block>>>(d_a, d_b, d_out, n);  // 3. launch
cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost); // 4. copy out
cudaFree(d_a);                                           // 5. free
```

!!! warning "Kernel launches are asynchronous and swallow errors silently"
    `<<<>>>` returns *immediately* — before the kernel runs. A bad launch config
    or an in-kernel fault won't throw; it sets a sticky error you must check:

    ```cpp
    kernel<<<g, b>>>(...);
    CUDA_CHECK(cudaGetLastError());     // launch errors (bad config)
    CUDA_CHECK(cudaDeviceSynchronize()); // execution errors (after running)
    ```

    The repo's `CUDA_CHECK` / `CUDA_CHECK_KERNEL` macros (`common/dojo/cuda_utils.cuh`)
    do this for you. Also run `compute-sanitizer ./your_demo` — it's the modern
    `cuda-memcheck` and catches out-of-bounds and races the compiler can't.

## A first taste of cooperation: reduction

Vector add is *embarrassingly* parallel — no thread needs another. Summing an
array is different: the answer is a single number that depends on all of them.
The trick is a **tree** — pairs of threads combine, halving the active set each
step, finishing in log₂N instead of N. Step through it:

<div data-dojo="reduction"></div>

The repo's `reduce_sum` does exactly this inside each block (shared memory +
`__syncthreads()`), then sums the per-block partials. We'll rebuild it properly
with warp primitives at [Level 6](level06-warps.md).

## Your reps

- Build and run: `cmake --build build && ctest --test-dir build`
- Read all three kernels in `levels/level01_vector_add/` — they're commented to
  match this page.
- **Then extend:** add an `image_invert` kernel (`out = 255 - in`) over a `uint8`
  buffer. It's vector add in disguise, and it's your bridge to
  [Level 2](level02-thread-mapping.md).

??? question "Self-check"
    1. You launch `<<<10, 256>>>` on `n = 3000`. How many threads ran? How many
       did real work? *(2560 launched; with the grid-stride loop all cover 3000
       by striding; with a plain bounds guard, 3000 do work and the rest idle —
       but 2560 < 3000, so a guard-only kernel would miss elements. This is why
       grid-stride or `ceil_div` sizing matters.)*
    2. Why is `cudaMemcpy` after the launch safe without an explicit sync?
       *(Default-stream copies wait for prior default-stream work — the copy-out
       implicitly synchronizes. Not true once you use streams: Level 8.)*

→ Continue to [Level 2 — Thread Mapping](level02-thread-mapping.md)
