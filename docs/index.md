<p align="center">
  <img src="assets/dojo-logo.webp" alt="CUDA Dojo logo" width="300" />
</p>

# CUDA Dojo

> *"What I cannot create, I do not understand."* — Richard Feynman

This is the companion guide to the [`cuda-dojo`](https://github.com/) code repo. It
is **not** a reference manual. Each page takes one idea, builds it up from
something you already know, and gives you an interactive toy to poke at until the
idea clicks. Then you go write the kernel.

The whole site is built around one belief: **you understand a GPU concept when you
can predict what the widget will do before you move the slider.** So on every
page, read the question, *guess the answer*, then check yourself against the
interactive example.

## How to use this guide

1. **Read the hook.** Each level opens with a question you can't yet answer.
2. **Build intuition.** We reason from CPUs, SIMD, and physics — no magic.
3. **Play.** Move the sliders. Try to break your own mental model.
4. **Do the reps.** Each level maps to a project in the code repo.
5. **Measure.** From Level 4 on, you stop guessing and start profiling.

## The skill tree

| Level | The one idea | Project |
|------:|--------------|---------|
| [0](level00-mental-model.md) | A GPU trades latency for throughput | vector add / SAXPY / reduction |
| [1](level01-basics.md) | A kernel is one function run by a grid of threads | invert / threshold |
| [2](level02-thread-mapping.md) | You design the thread→data mapping | transpose / crop / resize |
| [3](level03-memory-hierarchy.md) | Where data lives dominates speed | tiled transpose |
| [4](level04-performance.md) | Every kernel is memory- or compute-bound | optimize blur / Sobel |
| [5](level05-shared-memory.md) | A block is a team with a shared scratchpad | convolution / box filter |
| [6](level06-warps.md) | 32 threads can talk without memory | warp reduction / scan |
| [7](level07-libraries.md) | Don't hand-roll GEMM or FFT | cuBLAS / cuFFT pipeline |
| [8](level08-streams.md) | Overlap copy and compute | video pipeline |
| [9](level09-multi-kernel.md) | Real programs are kernel graphs | GEMM |
| [10](level10-optimization.md) | Nsight tells you the truth | 20 ms → 1 ms |
| [11](level11-advanced.md) | Tensor cores, graphs, multi-GPU | — |
| [12](level12-architecture.md) | Production = pools + graphs + streams | end-to-end pipeline |
| [13](level13-algorithm-design.md) | Reformulate the algorithm for the hardware | your own |

## Tailored path

Coming from HPC / SIMD / OpenMP / image processing, blast through Levels 0–2 —
they'll feel familiar. The real ROI is **Levels 4–10**: memory hierarchy, warp
programming, Nsight-driven optimization, and stream/graph pipelines. That's where
high-throughput imaging and signal-processing performance is actually won.
