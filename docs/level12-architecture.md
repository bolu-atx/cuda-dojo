# Level 12 — Production Architecture

> **The question:** *A kernel is a function. A product is a system that runs that
> function a million times a second under a latency budget, sharing a GPU with
> other work. What surrounds the kernels?*

You now think beyond individual kernels to the system that orchestrates them. The
shape of a real high-throughput GPU application:

```
        Application
            │
        Scheduler            (what runs when, on which stream)
            │
        Memory Pool          (reuse allocations — never cudaMalloc in the hot loop)
            │
        CUDA Streams         (overlap copy / compute / copy)
            │
        CUDA Graph           (replay the kernel sequence with one launch)
            │
  cuBLAS · cuFFT · custom kernels · tensor cores
            │
           GPU
```

Every layer here you've met as an isolated idea (Levels 7–11). Production is
assembling them into something maintainable and fast.

## Memory pools — the #1 production rule

`cudaMalloc`/`cudaFree` are **synchronous and slow** — they can stall the whole
device. Calling them in your steady-state loop destroys throughput. Production code
allocates *once* and reuses:

- A **memory pool / arena** (or `cudaMallocAsync` with a `cudaMemPool_t`) hands out
  buffers from a pre-reserved region with near-zero overhead.
- Frameworks (PyTorch's caching allocator, RAPIDS RMM) are exactly this. The rule:
  **never allocate in the hot path.**

## Steady-state pipeline = streams + graphs

Combine Level 8 (overlap) and Level 11 (graphs): build the pipeline once as a
stream graph, capture it, then replay per frame/batch. Transfers hide behind
compute and launch overhead is amortized to ~one call:

<div data-dojo="streams"></div>

For a video/imaging service this is the difference between hitting frame rate and
not: at steady state, copy-in and copy-out are *invisible*, fully overlapped with
the kernels.

## The data-movement hierarchy you must respect

Performance at the system level is dominated by *where data lives and how it
travels*:

| Path | Rough bandwidth | Lesson |
|------|-----------------|--------|
| GPU global memory | ~1–3 TB/s | the fast lane — stay here |
| NVLink (GPU↔GPU) | ~100s GB/s | far better than PCIe for multi-GPU |
| PCIe (host↔GPU) | ~tens GB/s | the bottleneck — minimize crossings |
| GPUDirect (NIC/storage↔GPU) | bypasses CPU | skip the host bounce entirely |

- **Zero-copy / pinned / unified memory** — choose deliberately; each trades
  control for convenience.
- **GPUDirect RDMA / Storage** — move data NIC→GPU or NVMe→GPU without staging
  through host RAM. Essential for high-ingest pipelines (think streaming sensor or
  sequencing data).
- **NUMA awareness** — pin host threads/memory to the socket nearest the GPU's PCIe
  root; cross-socket transfers are measurably slower.

!!! tip "The mental shift"
    At kernel level you optimize *cycles*. At system level you optimize *data
    movement and scheduling*. The fastest kernel in the world loses to a pipeline
    that round-trips over PCIe between every stage. Architecture is mostly about
    keeping bytes on the GPU and the engines never idle.

## Your reps

- **End-to-end image pipeline** — ingest → preprocess → inference (cuBLAS/cuDNN) →
  postprocess → output, with: a memory pool, multi-stream overlap, a captured CUDA
  graph for the per-frame path, and no allocations in the loop. Measure sustained
  throughput and tail latency, not just kernel time.
- Profile the *whole system* in `nsys`: prove transfers are hidden and there are no
  gaps between graph replays.

→ Continue to [Level 13 — Algorithm Design](level13-algorithm-design.md)
