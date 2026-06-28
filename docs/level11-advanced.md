# Level 11 — Advanced CUDA

> **The question:** *You've mastered the standard model — grids, blocks, warps,
> shared memory. What capabilities exist beyond it, and when is each one actually
> the right tool?*

These are specialized features. Don't reach for them by default — each adds
complexity and most kernels never need them. But knowing they exist (and their
trade-offs) is what separates "can write CUDA" from "architects CUDA."

## Tensor Cores & WMMA — the matmul accelerators

Separate hardware units that do a small **matrix multiply-accumulate** in one
operation (e.g. 16×16×16), at many times the throughput of the regular FP32 ALUs.
They power modern AI. You rarely program them directly — you get them through
**cuBLAS / cuBLASLt / CUTLASS / cuDNN**. The `wmma` API exists if you must fuse a
custom op into a matmul, but the bar for hand-writing them is high.

- Trade-off: reduced precision inputs (FP16/BF16/TF32/FP8) accumulating in FP32.
  Great for DL and increasingly for HPC mixed-precision; verify your numerics.
- cuBLASLt is the modern GEMM interface when you need epilogues such as bias,
  activation, or autotuned layouts.

This pushes the *compute ceiling* of the roofline way up — but only for the
matmul-shaped work that can use them:

<div data-dojo="roofline"></div>

## CUDA Graphs — kill launch overhead

Record a whole sequence of kernels + copies once, then **replay** it with a single
launch. When you have many small kernels in a tight loop (Level 9's launch-overhead
problem), graphs can cut CPU dispatch cost dramatically.

```cpp
cudaStreamBeginCapture(stream, ...);   // record the sequence
//   ... your normal launches & async copies ...
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&exec, graph, ...);
cudaGraphLaunch(exec, stream);         // replay: one call, whole pipeline
```

The natural successor to Level 8 streams for *steady-state* pipelines.

## Cooperative Groups — flexible synchronization

A modern API that generalizes `__syncthreads()` to explicit, composable groups:
sub-warp tiles, the whole block, or — with a cooperative launch — the **entire
grid**. Grid-wide sync lets some algorithms that needed multiple kernels (Level 9)
run as one persistent kernel.

```cpp
namespace cg = cooperative_groups;
auto block = cg::this_thread_block();
auto warp  = cg::tiled_partition<32>(block);
warp.sync();                 // clearer than raw masks
auto grid  = cg::this_grid();
grid.sync();                 // grid-wide barrier (cooperative launch required)
```

## Persistent & dynamic-parallelism kernels

- **Persistent kernels** — launch exactly enough blocks to fill the GPU, then have
  them loop pulling work from a queue. Eliminates per-iteration launch overhead;
  the backbone of some low-latency and producer/consumer designs.
- **Dynamic parallelism** — a kernel launches *child* kernels from the device.
  Elegant for recursive/adaptive work (tree traversal, adaptive mesh), but the
  overhead is real — profile before committing.

## Multi-GPU: NCCL, NVSHMEM, MIG

- **NCCL** — optimized collectives (all-reduce, broadcast) across GPUs/nodes; the
  standard for scaling DL training. Use it; don't hand-roll cross-GPU reductions.
- **NVSHMEM** — a PGAS model: GPUs directly read/write each other's memory.
- **MIG** — partition one big GPU into isolated instances for multi-tenant serving.

!!! warning "Advanced ≠ default"
    Every feature here trades simplicity for a specific gain. Reach for them when a
    profiler points at the exact problem they solve (launch overhead → graphs;
    matmul ceiling → tensor cores; cross-GPU scale → NCCL) — not because they're
    advanced.

## Your reps

- Wrap a small multi-kernel loop in a **CUDA Graph**; measure the launch-overhead
  reduction with `nsys`.
- Rewrite a Level 6 warp reduction using **cooperative groups** tiles and compare
  clarity.
- If you have ≥2 GPUs: an **NCCL all-reduce** of a vector, and reason about the
  bandwidth across NVLink/PCIe.
- Use the [Library Field Guide](track-libraries.md) before trying to write WMMA or
  CUTLASS code from scratch.

→ Continue to [Level 12 — Production Architecture](level12-architecture.md)
