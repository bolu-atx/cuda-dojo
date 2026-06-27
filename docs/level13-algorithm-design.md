# Level 13 — Algorithm Design

> **The question:** *You stop asking "how do I parallelize this loop?" and start
> asking "what algorithm maps naturally onto warps, blocks, and the memory
> hierarchy?" When is the answer a* different *algorithm entirely?*

This is the summit. You're no longer *writing CUDA* — you're *designing GPU-native
algorithms*. The best GPU code often isn't the parallelized version of the CPU
algorithm; it's a reformulation that the hardware likes.

## The reframe

> ❌ *How do I parallelize this algorithm?*
> ✅ *How should this computation be restructured so work maps onto warps and
> blocks, and data flows up the memory hierarchy?*

The CPU-optimal algorithm assumes cheap random access, deep caches, and a few fast
threads. The GPU offers the opposite: thousands of slow threads, expensive
scattered access, fast contiguous access, and a tiny per-block scratchpad. A
different cost model demands a different algorithm.

## The recurring design patterns

### Tiling instead of nested loops
Don't stream over global memory repeatedly. Partition the problem into tiles that
fit in shared memory, do all the reuse on-chip, move to the next tile. (Transpose,
convolution, GEMM — all the same shape.)

### Hierarchical combine instead of global atomics
A million threads doing `atomicAdd` to one global counter serializes into a queue.
Instead: combine **within the warp** (shuffles), then **within the block** (shared
memory), then one atomic **per block** to global. The reduction tree you stepped
through is this principle in miniature — push the combining as far up the hierarchy
as possible before touching slow shared resources:

<div data-dojo="reduction"></div>

### Privatize, then merge
Histograms: don't let all threads fight over global bins. Give each block its own
private copy in shared memory, fill it conflict-free, then merge block-private
results into global once. (Level 5/7.) Same idea as hierarchical combine.

### Structured access instead of scatter/gather
Random scatter is the GPU's worst case (uncoalesced, atomic-heavy). Reformulate so
threads write *contiguous* runs: **sort by destination then segment-reduce**, or
build offsets via a prefix scan and write in order. Turning a scatter into a
sort+scan is a signature GPU move.

### Producer/consumer overlap instead of phases
Don't do "load everything, then compute everything." Pipeline so compute on tile
*N* overlaps the load of tile *N+1* (Levels 8/12). The algorithm is designed for
overlap from the start, not retrofitted.

## A worked reframe: counting

**CPU instinct:** loop, `++count[key]` — random writes, fine with caches.

**GPU-native:** that's a scatter into contention. Reframe as **sort the keys, then
the count of each key is the length of its run** — a segmented reduction over
contiguous, coalesced memory using library `sort` + `reduce_by_key`. Same answer,
but every access is now the hardware's *best* case instead of its worst. That's the
whole skill of this level: recognizing when the GPU-friendly algorithm is a
*different* algorithm.

!!! tip "The design checklist for any new GPU algorithm"
    1. **Is the work regular?** Irregular work → restructure (sort, bin, compact)
       until it is.
    2. **Is access coalesced?** If threads scatter, can a sort/scan make it
       contiguous?
    3. **What's the reuse factor?** High → tile into shared memory. Low → it's
       memory-bound; minimize bytes moved.
    4. **Where can combining happen?** Push reductions up: warp → block → global.
    5. **Can stages overlap?** Design the pipeline for producer/consumer overlap
       up front.

## Your reps

- Take an algorithm from *your* domain (a bioinformatics or signal-processing
  step) and run it through the checklist. Write down the CPU formulation and the
  GPU-native reformulation *before* coding.
- Re-derive: why is "sort + segmented reduce" usually faster than atomics for
  building a histogram with many keys, even though it does *more* total work?
  *(Because it converts contention + scatter into coalesced, contention-free
  passes — and on a bandwidth-bound machine, regular work beats less-but-irregular
  work.)*

---

You've reached the end of the tree. From here it's reps: take each project in the
code repo, drive it with the profiler, and — most importantly — start reformulating
*your own* problems to fit the machine. That's the difference between using a GPU
and thinking in one.
