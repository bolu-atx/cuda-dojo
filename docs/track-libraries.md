# Library Field Guide

> **The question:** *Is the CUDA skill here writing a kernel, or recognizing that
> the kernel should not exist?*

Senior CUDA engineers write fewer kernels than beginners. They know when a
primitive is already solved, and they spend custom-kernel effort only where the
application is genuinely custom.

Make a prediction: given the workload shape, which tool should own it?

<div data-dojo="library-choice"></div>

## The decision ladder

```
standard primitive?
├── reduction / scan / sort / histogram  →  CUB first, Thrust for prototypes
├── dense matrix multiply                →  cuBLAS or cuBLASLt
├── FFT / spectral method                →  cuFFT
├── common image operation               →  NPP / CV-CUDA / OpenCV CUDA
├── neural-network primitive             →  cuDNN / TensorRT
└── unusual shape, fusion, or semantics  →  custom CUDA or CUTLASS
```

The Feynman test is simple: before coding, explain why the library does not fit.
If you cannot, call the library and measure.

## Prototype high, optimize low

**Thrust** is the GPU STL: sort, transform, reduce, scan, copy-if. It is excellent
for expressing an idea quickly.

**CUB** is the production primitive box: reductions, scans, radix sort,
histograms, and block/warp collectives. If you are hand-writing one of these,
ask what special constraint CUB cannot satisfy.

## Math libraries are performance ceilings

cuBLAS and cuFFT represent years of architecture-specific tuning. Use them as the
baseline before trusting custom code:

- GEMM, GEMV, batched GEMM, and strided batched GEMM belong to cuBLAS/cuBLASLt.
- FFT convolution, deconvolution, registration, and frequency filtering belong to
  cuFFT plus small custom kernels between transforms.
- Tensor Core paths usually come through cuBLASLt or CUTLASS first.

The custom work is often the glue: layout conversion, normalization, masking,
boundary handling, or fusing memory-bound stages around the library call.

## Image primitives are libraries too

NPP, CV-CUDA, and OpenCV CUDA cover many production image operations: resize,
rotate, color conversion, morphology, filtering, thresholding, histograms, and
arithmetic. A custom kernel is still valuable for learning, but production code
should compare semantics and throughput against these primitives.

## Your reps

- Replace a handwritten reduction with CUB and predict which code gets simpler.
- Build an FFT convolution pipeline with cuFFT; keep only the frequency-domain
  multiply as custom CUDA.
- Compare a custom image primitive against NPP. Check boundary behavior before
  comparing speed.

??? question "Self-check"
    You wrote a GEMM that reaches 40% of peak, while cuBLAS reaches 90%+. What is
    the production move? *(Use cuBLAS/cuBLASLt unless your shape, datatype, or
    fusion requirement is truly outside the library.)*
