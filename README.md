# CUDA Dojo

A level-by-level CUDA learning project, structured as a skill tree rather than a
textbook. Each level unlocks one mental model and one production skill. Host code
is **C++23**, device code is **C++20** (the highest CUDA reliably supports for
`__device__` code today).

> **Hardware:** CUDA requires an NVIDIA GPU. macOS/Apple Silicon cannot build or
> run this — develop on a Linux/Windows box with the CUDA Toolkit installed.

## Build & test

```bash
cmake -B build -G Ninja          # configure (targets the local GPU arch)
cmake --build build              # build everything
ctest --test-dir build --output-on-failure   # run all level tests
./build/levels/level01_vector_add/level01_demo   # run a level's demo
```

Pin specific architectures instead of autodetecting:

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES="80;86;90"
```

## Project layout

```
common/dojo/cuda_utils.cuh   CUDA_CHECK, GpuTimer, bandwidth/FLOP helpers, device info
common/dojo/test.hpp         zero-dependency micro test harness (CTest-backed)
cmake/Dojo.cmake             add_dojo_level() — one helper builds lib + demo + test
levels/levelNN_<topic>/      each level: kernels (.cu), a demo, and a test
```

### Anatomy of a level

Kernels live in a small static library that **both** the demo and the test link
against, so the canonical host+kernel code is written once and exercised two
ways. Add a new level by creating `levels/levelNN_<topic>/` with a
`CMakeLists.txt` calling `add_dojo_level(...)`, then `add_subdirectory(...)` it
from the top-level `CMakeLists.txt`.

## The skill tree

| Level | Project | Core concepts | Status |
|------:|---------|---------------|:------:|
| 1 | Vector add / SAXPY / reduction | thread indexing, launches, `cudaMalloc`/`Memcpy`, error checking | ✅ |
| 2 | Image invert / threshold | 2D grids, thread→pixel mapping, boundary conditions | ⬜ |
| 3 | Matrix transpose | coalescing | ⬜ |
| 4 | Tiled transpose | shared memory, bank conflicts | ⬜ |
| 5 | Convolution | halo loading, `__syncthreads()` | ⬜ |
| 6 | Reduction (warp) | `__shfl_sync`, warp primitives | ⬜ |
| 7 | Histogram | atomics, privatization | ⬜ |
| 8 | Prefix scan | cooperative algorithms | ⬜ |
| 9 | GEMM | tiling, register blocking | ⬜ |
| 10 | FFT | multi-stage, cuFFT integration | ⬜ |
| 11 | Video pipeline | streams, overlap, CUDA Graphs | ⬜ |
| 12 | End-to-end image pipeline | production architecture | ⬜ |

Given an HPC/SIMD/OpenMP background, levels 1–2 should go fast; the real payoff
is levels 4–10 (memory hierarchy, warp programming, Nsight-driven perf analysis,
and stream/graph pipelines).

## Profiling (from Level 4 onward)

Release builds compile with `-lineinfo` so the profilers map SASS back to source:

```bash
nsys profile ./build/levels/.../levelNN_demo     # timeline: transfers, kernels, gaps
ncu --set full ./build/levels/.../levelNN_demo    # per-kernel: occupancy, memory, roofline
```
