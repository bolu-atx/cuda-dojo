# Imaging & Computer Vision

> **The question:** *Is this image operation a stencil, a transform, a reduction,
> or a pipeline?*

Images are a good CUDA teacher because pixels make the data movement visible.
Most imaging kernels are memory-bound, so the win is rarely "do less arithmetic."
The win is coalescing, reuse, separability, and keeping frames on the GPU.

Pick an algorithm and predict the shape of the GPU solution before moving the
controls.

<div data-dojo="imaging-pattern"></div>

## The pattern map

| Image task | GPU pattern | First optimization question |
|------------|-------------|-----------------------------|
| Sobel / Scharr / blur | stencil | can a tile reuse neighboring pixels? |
| Gaussian blur | separable stencil | can k×k become k + k? |
| Morphology | windowed min/max | can shared memory stage the neighborhood? |
| FFT convolution | spectral transform | is the kernel large enough for FFT to win? |
| Affine / perspective warp | gather with interpolation | are reads cached and writes coalesced? |
| Histogram / integral image | privatize + reduce / scan | where can combining happen before global memory? |
| Non-maximum suppression | local compare + compact | can scan turn sparse writes into contiguous writes? |

## Direct, separable, or FFT?

A small 3×3 Sobel filter is a stencil: each output reads a small neighborhood.
Shared-memory tiling can reduce duplicate loads.

A Gaussian blur has a stronger trick: separability. A k×k filter can become one
horizontal k-tap pass and one vertical k-tap pass. That changes the work from
roughly k² reads per pixel to 2k.

For very large kernels, FFT convolution can win because convolution becomes
multiply in the frequency domain. The cost shifts to cuFFT plans, complex layout,
batched transforms, and keeping the pipeline on the GPU.

## Pipelines beat isolated kernels

A production imaging flow is a graph:

```
ingest -> normalize -> filter/warp -> analyze/infer -> postprocess -> output
```

The pipeline question is whether copy engines, SMs, and libraries overlap. Video
decode/encode, TensorRT inference, and GPUDirect belong here as system pieces, not
as a separate beginner track.

## Your reps

- Implement Gaussian blur directly and separably. Predict the work ratio before
  timing.
- Build FFT convolution and find the kernel size where it beats spatial
  convolution.
- Compare custom morphology with NPP or CV-CUDA for semantics and speed.
- Mark a full imaging pipeline with NVTX ranges and inspect it in Nsight Systems.

??? question "Self-check"
    Why can a separable Gaussian blur beat a direct 2D blur by more than a small
    constant factor? *(It changes the algorithm: k² neighborhood work becomes two
    k-length passes, while memory access remains regular.)*
