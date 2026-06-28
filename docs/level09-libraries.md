# Level 9 — CUDA Libraries

> **The question:** *You could spend six months writing a matrix-multiply that
> reaches 90% of peak. NVIDIA already did. When should you write a kernel, and
> when should you call one?*

A senior CUDA engineer writes *fewer* kernels than a beginner. The skill here is
knowing the ecosystem so you only hand-roll what's genuinely custom, and let
battle-tested libraries own the hard, well-studied primitives.

## The toolbox

| Library | Owns | Reach for it when |
|---------|------|-------------------|
| **Thrust** | STL-like algorithms (`sort`, `reduce`, `scan`, `transform`) | prototyping, host-side orchestration |
| **CUB** | building blocks (block/warp/device reduce, scan, sort, histogram) | you need a fast primitive *inside* your kernel |
| **cuBLAS** | dense linear algebra (GEMM, GEMV) | any matrix multiply — don't write your own in prod |
| **cuFFT** | fast Fourier transforms | spectral methods, convolution-via-FFT |
| **cuRAND** | RNG (host and device) | Monte Carlo, augmentation, init |
| **cuSPARSE / cuSOLVER** | sparse matrices / decompositions | graphs, finite element, least squares, SVD/eigen |
| **NPP** | image primitives | resize, filters, morphology, histograms, color conversion |
| **cuDNN / CUTLASS / cuBLASLt** | DL primitives / templated GEMM / modern GEMM | specialized, or when you must fuse into GEMM |

!!! tip "Thrust vs CUB — the practical split"
    **Thrust** is high-level and host-driven: `thrust::reduce(d_vec.begin(), …)`
    launches kernels for you. Great for glue and prototypes. **CUB** is
    lower-level and composable: `cub::BlockReduce` is a *device* primitive you drop
    *inside* your own kernel. Rule of thumb: prototype with Thrust, optimize hot
    paths with CUB, then only write raw kernels for the parts no library covers.

## Real CUDA is a pipeline of kernels and library calls

Production code rarely looks like one heroic kernel. It looks like:

```
your custom preprocess kernel
        │
        ▼
   cuBLAS GEMM  ──►  cuFFT  ──►  your custom postprocess kernel
```

Each library call is just a kernel launch on the same device pointers, on the same
stream. The libraries interoperate because they all speak raw `cudaMalloc`'d
pointers and `cudaStream_t`.

## Why this matters even for performance

GEMM is **compute-bound** (high arithmetic intensity) — far right on the roofline.
Hitting the compute ceiling there requires register blocking, double buffering,
and often tensor cores: thousands of engineer-hours. cuBLAS/CUTLASS already sit at
that ceiling:

<div data-dojo="roofline"></div>

Your value-add is the *custom* memory-bound glue around those calls — fusing your
pre/post-processing so you don't round-trip through DRAM between stages (a Level 11
theme).

## Your reps

- **GEMM pipeline** — implement a naive tiled GEMM yourself (Level 11), then call
  cuBLAS on the same matrices and measure the gap. Feel how far "good enough hand
  code" is from the library. Humbling and clarifying.
- **FFT pipeline** — `cufftPlan2d` → forward FFT → your custom frequency-domain
  filter kernel → inverse FFT. This is FFT-based convolution, a real imaging
  technique.
- Replace your Level 1 reduction with **`cub::DeviceReduce::Sum`** and compare
  lines of code and throughput.
- Try one **NPP image primitive** that overlaps with a custom exercise. Compare
  boundary semantics first, then speed.

For the broader production map, continue into the
[Library Field Guide](track-libraries.md).

??? question "Self-check"
    You wrote a tuned GEMM hitting 40% of peak FLOPS. cuBLAS hits 92%. Should you
    keep optimizing yours? *(Almost never for standard dense GEMM — call cuBLAS or
    CUTLASS. Spend your effort on the memory-bound custom kernels around it, and on
    *fusing* them to cut DRAM round-trips. Hand-rolled GEMM is justified only for
    unusual shapes/dtypes/fusions the libraries don't cover.)*

→ Continue to [Level 10 — Streams & Asynchrony](level10-streams.md)
