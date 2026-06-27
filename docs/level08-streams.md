# Level 8 — Streams & Asynchrony

> **The question:** *Your GPU has separate engines for copy-in, compute, and
> copy-out. In the naive flow, two of the three sit idle at all times. How do you
> run all three at once?*

So far everything has been serial: copy all data in, run the kernel, copy all
results out. The GPU has *dedicated DMA engines* for transfers that can run
concurrently with the compute cores — but only if you stop forcing them to wait
for each other. That's what streams are for.

## A stream is a queue

A **stream** is an ordered queue of operations. Work in the *same* stream runs in
order; work in *different* streams can overlap. The default stream (stream 0) is
the synchronization bottleneck you've been using implicitly.

Two ingredients unlock overlap:

1. **Pinned (page-locked) host memory** (`cudaMallocHost`) — async copies require
   it; pageable memory silently falls back to synchronous.
2. **`cudaMemcpyAsync` + a non-default stream** — so the copy returns immediately
   and the engine runs alongside compute.

```cpp
for (int i = 0; i < nChunks; ++i) {
    int s = i % nStreams;
    cudaMemcpyAsync(d_in + off, h_in + off, bytes, H2D, stream[s]);
    kernel<<<g, b, 0, stream[s]>>>(d_in + off, d_out + off, n);
    cudaMemcpyAsync(h_out + off, d_out + off, bytes, D2H, stream[s]);
}
```

## See the overlap

Chop the work into chunks and pipeline them across streams: while chunk 1
computes, chunk 2 is already copying in. Drag the chunk count and watch the
overlapped timeline collapse toward the single-engine bound:

<div data-dojo="streams"></div>

The serial timeline is `chunks × 3` stages long. The overlapped one is
`chunks + 2` — the +2 is just the pipeline fill/drain. More chunks ⇒ the fixed
overhead amortizes and you approach a **3× speedup** (three engines fully busy).

!!! warning "Overlap requires independence and pinned memory"
    - Copies and kernels overlap only across *different* streams **and** only if
      the host memory is pinned. Forget `cudaMallocHost` and you'll see no overlap
      and wonder why.
    - Don't sprinkle `cudaDeviceSynchronize()` — it serializes everything. Use
      **events** (`cudaEventRecord` / `cudaStreamWaitEvent`) for fine-grained
      cross-stream dependencies, and `cudaStreamSynchronize(stream)` to wait on
      just one.

## Events: timing and dependencies

You already use `cudaEvent_t` for timing (the repo's `GpuTimer`). The same events
express **dependencies**: `cudaStreamWaitEvent(streamB, evt)` makes stream B wait
for a point in stream A — a producer/consumer handoff without stalling the host.

## Your reps

- **Chunked vector add / SAXPY** — take Level 1's kernel, pin the host buffers,
  split into N chunks across 3–4 streams, and measure wall-clock vs the serial
  version. Confirm the speedup the widget predicts.
- **Video / camera pipeline** — decode/upload frame *N+1* while processing frame
  *N* and downloading frame *N−1*. This is the real-world payoff: a steady-state
  pipeline where transfers are completely hidden.
- Profile with `nsys` — the timeline view *shows* the overlapping bars; this is
  the most satisfying profiler picture in CUDA.

??? question "Self-check"
    You pipeline across 8 chunks and 3 streams but see no speedup in `nsys`. Name
    the two most likely causes. *(1) Host memory isn't pinned, so `cudaMemcpyAsync`
    ran synchronously. (2) A stray `cudaDeviceSynchronize()` or use of the default
    stream between iterations serialized the queues.)*

→ Continue to [Level 9 — Multi-Kernel Algorithms](level09-multi-kernel.md)
