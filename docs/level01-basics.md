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

<svg class="dojo-diagram" viewBox="0 0 760 210" role="img" aria-label="A grid contains blocks; each block contains threads.">
  <rect class="stroke-faint" x="10" y="12" width="740" height="60" rx="8"/>
  <text class="mono" x="20" y="30">grid — gridDim.x = 4 blocks</text>
  <g class="mono">
    <rect class="fill-faint" x="30"  y="38" width="150" height="26" rx="4"/><text x="105" y="55" text-anchor="middle">block 0</text>
    <rect class="fill-faint" x="210" y="38" width="150" height="26" rx="4"/><text x="285" y="55" text-anchor="middle">block 1</text>
    <rect class="fill-accent" x="390" y="38" width="150" height="26" rx="4"/><text x="465" y="55" text-anchor="middle" fill="#0b1500">block 2</text>
    <rect class="fill-faint" x="570" y="38" width="150" height="26" rx="4"/><text x="645" y="55" text-anchor="middle">block 3</text>
  </g>
  <path class="stroke-faint" stroke-dasharray="4 4" d="M390,64 L30,120 M540,64 L750,120"/>
  <rect class="stroke-accent" x="10" y="120" width="740" height="78" rx="8"/>
  <text class="mono" x="20" y="138">block 2 — blockDim.x = 8 threads (threadIdx.x = 0…7)</text>
  <g class="mono">
    <!-- 8 thread cells -->
    <g>
      <rect class="fill-faint" x="30"  y="150" width="86" height="34" rx="4"/><text x="73"  y="172" text-anchor="middle">t0</text>
      <rect class="fill-faint" x="120" y="150" width="86" height="34" rx="4"/><text x="163" y="172" text-anchor="middle">t1</text>
      <rect class="fill-faint" x="210" y="150" width="86" height="34" rx="4"/><text x="253" y="172" text-anchor="middle">t2</text>
      <rect class="fill-accent" x="300" y="150" width="86" height="34" rx="4"/><text x="343" y="172" text-anchor="middle" fill="#0b1500">t3 → idx 19</text>
      <rect class="fill-faint" x="390" y="150" width="86" height="34" rx="4"/><text x="433" y="172" text-anchor="middle">t4</text>
      <rect class="fill-faint" x="480" y="150" width="86" height="34" rx="4"/><text x="523" y="172" text-anchor="middle">t5</text>
      <rect class="fill-faint" x="570" y="150" width="86" height="34" rx="4"/><text x="613" y="172" text-anchor="middle">t6</text>
      <rect class="fill-faint" x="660" y="150" width="86" height="34" rx="4"/><text x="703" y="172" text-anchor="middle">t7</text>
    </g>
  </g>
</svg>

The grid is the unit of scale; the block is the unit of cooperation; the thread
is the unit of work. Element 19 lives at `blockIdx 2 · blockDim 8 + threadIdx 3`.

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
    scratchpad and synchronize (Level 6). Blocks are the unit of cooperation;
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

The GPU can't touch your `std::vector`. Host RAM and the RTX PRO 6000's 96 GB of
GDDR7 are separate address spaces — every input has to be *driven across the PCIe
bridge* before a thread can read it, and results driven back:

<svg class="dojo-diagram" viewBox="0 0 760 170" role="img" aria-label="Host RAM and device GDDR7 are separate; data is copied across PCIe.">
  <rect class="stroke-faint" x="10" y="40" width="240" height="90" rx="8"/>
  <text class="mono" x="130" y="30" text-anchor="middle">HOST (CPU RAM)</text>
  <text class="mono" x="130" y="78" text-anchor="middle">h_a, h_b, h_out</text>
  <text class="mono" x="130" y="100" text-anchor="middle">std::vector&lt;float&gt;</text>
  <rect class="stroke-accent" x="510" y="40" width="240" height="90" rx="8"/>
  <text class="mono" x="630" y="30" text-anchor="middle">DEVICE (96 GB GDDR7)</text>
  <text class="mono" x="630" y="78" text-anchor="middle">d_a, d_b, d_out</text>
  <text class="mono" x="630" y="100" text-anchor="middle">cudaMalloc</text>
  <path class="stroke-accent" stroke-width="2" marker-end="url(#dojoarrow)" d="M250,68 L505,68"/>
  <path class="stroke-warn"   stroke-width="2" marker-end="url(#dojoarrow2)" d="M505,108 L250,108"/>
  <text class="mono fill-accent" x="378" y="60" text-anchor="middle">cudaMemcpy …HostToDevice (1)</text>
  <text class="mono fill-warn" x="378" y="126" text-anchor="middle">cudaMemcpy …DeviceToHost (3)</text>
  <text class="mono" x="630" y="150" text-anchor="middle">kernel runs here (2)</text>
  <defs>
    <marker id="dojoarrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path class="fill-accent" d="M0,0 L6,3 L0,6 z"/></marker>
    <marker id="dojoarrow2" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="#e0633a"/></marker>
  </defs>
</svg>

The canonical five-step flow (this is exactly `vector_add` in
`levels/level01_vector_add/vector_add.cu`):

=== "Host: the five-step dance"

    ```cpp { .annotate }
    cudaMalloc(&d_a, bytes);                                  // (1)!
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);      // (2)!
    vector_add_kernel<<<grid, block>>>(d_a, d_b, d_out, n);   // (3)!
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);  // (4)!
    cudaFree(d_a);                                            // (5)!
    ```

    1.  **Allocate on the device.** `d_a` is a *device* pointer — a GDDR7 address.
        Dereferencing it on the host is a crash; it's only valid inside the kernel.
    2.  **Copy in.** Drive the input bytes across PCIe, host → device. Nothing on
        the GPU exists until you put it there.
    3.  **Launch.** `<<<grid, block>>>` runs the kernel across the whole grid. It
        returns *immediately* — see the warning below.
    4.  **Copy out.** Drive the results back, device → host, so the CPU can see them.
    5.  **Free.** Device memory is not garbage-collected. Leak it and it stays gone
        until the process exits.

=== "Kernel"

    ```cpp { .annotate }
    __global__ void vector_add_kernel(const float* a, const float* b,
                                      float* out, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;     // (1)!
        if (idx < n)                                         // (2)!
            out[idx] = a[idx] + b[idx];                      // (3)!
    }
    ```

    1.  The one line that maps *this* thread to *its* element — the whole hierarchy
        flattened to a single number.
    2.  The boundary guard: extra threads (you launched ≥ n) must do nothing.
    3.  Pure data parallelism — no thread needs any other. That's what makes it
        embarrassingly parallel.

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
with warp primitives at [Level 7](level07-warps.md).

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
       implicitly synchronizes. Not true once you use streams: Level 10.)*

→ Continue to [Level 2 — Thread Mapping](level02-thread-mapping.md)
