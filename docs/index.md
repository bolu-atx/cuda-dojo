<p align="center">
  <img src="assets/dojo-logo.webp" alt="CUDA Dojo logo" width="300" />
</p>

# CUDA Dojo

> *"What I cannot create, I do not understand."* — Richard Feynman

This is the companion guide to the [`cuda-dojo`](https://github.com/bolu-atx/cuda-dojo) code repo. It
is **not** a reference manual. Each page takes one idea, builds it up from
something you already know, and gives you an interactive toy to poke at until the
idea clicks. Then you go write the kernel.

The whole site is built around one belief: **you understand a GPU concept when you
can predict what the widget will do before you move the slider.** So on every
page, read the question, *guess the answer*, then check yourself against the
interactive example.

## Why this still matters in the age of AI

An AI agent can emit a tiled transpose or a warp reduction in seconds — the
*tricks* are now cheap. What it cannot do for you is hold the mental model: knowing
that a GPU trades latency for throughput, that a kernel is memory- or compute-bound,
that where data lives dominates speed. Those big concepts are what let you *judge*
the code an agent hands you — to ask the right question, spot the kernel that quietly
serializes a warp, predict which wall you'll hit, and tell a real speedup from a
plausible-looking one.

The techniques follow from the foundation, not the other way around. Memorize a trick
and you can reproduce one kernel; understand the model and every trick becomes obvious,
including the ones nobody has written down yet. That is the part the agent can't
outsource — and the part this dojo trains. Feynman's line at the top is the whole
point: if you can only generate it, you don't yet understand it.

## How to use this guide

1. **Read the hook.** Each level opens with a question you can't yet answer.
2. **Build intuition.** We reason from CPUs, SIMD, and physics — no magic.
3. **Play.** Move the sliders. Try to break your own mental model.
4. **Do the reps.** Each level maps to a project in the code repo.
5. **Measure.** From Level 5 on, you stop guessing and start profiling.

## The skill tree

| Level | The one idea | Project |
|------:|--------------|---------|
| [0](level00-mental-model.md) | A GPU trades latency for throughput | vector add / SAXPY / reduction |
| [1](level01-basics.md) | A kernel is one function run by a grid of threads | invert / threshold |
| [2](level02-thread-mapping.md) | You design the thread→data mapping | transpose / crop / resize |
| [3](level03-execution-model.md) | CUDA has a logical machine and a physical machine | scope mapping drills |
| [4](level04-memory-hierarchy.md) | Where data lives dominates speed | tiled transpose |
| [5](level05-performance.md) | Every kernel is memory- or compute-bound | optimize blur / Sobel |
| [6](level06-shared-memory.md) | A block is a team with a shared scratchpad | box filter / separable blur |
| [7](level07-warps.md) | Warp lanes cooperate through masks and registers | warp reduction / histogram |
| [8](level08-synchronization-scopes.md) | Synchronization is a scope decision | warp/block/stream idioms |
| [9](level09-libraries.md) | Don't hand-roll GEMM or FFT | cuBLAS / cuFFT pipeline |
| [10](level10-streams.md) | Overlap copy and compute | video pipeline |
| [11](level11-multi-kernel.md) | Real programs are kernel graphs | GEMM |
| [12](level12-optimization.md) | Nsight tells you the truth | 20 ms → 1 ms |
| [13](level13-orchestration.md) | Compose work into pipelines with streams, events, graphs | producer/consumer pipeline |
| [14](level14-architecture.md) | Production = memory pools + data movement (PCIe / NVLink / GPUDirect) | end-to-end pipeline |
| [15](level15-algorithm-design.md) | Reformulate the algorithm for the hardware | your own |
| [16](level16-tile-pipeline.md) | Pipeline tiles inside one kernel | `cp.async` GEMM tile loop |

## Advanced tracks

The levels are the main climbing route. The tracks are the senior-engineer map:
use them once the core model is stable, or dip into one when a project demands it.

| Track | What it covers | When to study it |
|------|----------------|------------------|
| [Architecture Deep Dive](track-architecture.md) | SM anatomy, schedulers, issue slots, occupancy, cache behavior, tensor cores, multi-GPU (NCCL/NVSHMEM, MIG) | when profiler metrics need a hardware explanation, or you scale past one GPU |
| [Library Field Guide](track-libraries.md) | CUB, Thrust, cuBLAS, cuFFT, cuRAND, cuSPARSE, cuSOLVER, NPP | before writing a kernel for a standard primitive |
| [Imaging & Computer Vision](track-imaging.md) | stencils, separable filters, FFT convolution, image warps, production pipelines | for microscopy, cameras, medical imaging, and production CV |

## Tailored path

Coming from HPC / SIMD / OpenMP / image processing, blast through Levels 0–2 —
they'll feel familiar. The real ROI is **Levels 4–10**: memory hierarchy, warp
programming, Nsight-driven optimization, and stream/graph pipelines. That's where
high-throughput imaging and signal-processing performance is actually won.
