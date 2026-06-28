# Level 14 — The Intra-Kernel Pipeline (cp.async · TMA · Clusters)

> **The question:** *Your tiled GEMM (Level 9) loads a tile into shared memory, then
> computes on it, then loads the next. Load and compute take turns — so the math units
> sit idle during every load. How do you overlap the* next *tile's load with* this
> *tile's compute, all inside one kernel?*

This is the third scale of concurrency from [Level 11](level11-orchestration.md). Layer
4–6 pipelined work *across* streams; here you pipeline *within* a single kernel. It's the
same idea — keep every engine busy by staggering stages — pushed down to the level of
tiles moving through the memory hierarchy you mapped at [Level 3](level03-memory-hierarchy.md):
global → shared → registers → tensor cores.

This is also the most hardware-specific level in the dojo. `cp.async` needs Ampere (SM80);
TMA and clusters need Hopper (SM90) or newer. None of it can be verified on older silicon —
where a claim needs that hardware to confirm, this page says so.

## The serialization you're paying for

A classic shared-memory tile loop ([Level 5](level05-shared-memory.md),
[Level 9](level09-multi-kernel.md)) does this per K-step:

```cpp
load_tile_to_shared(k);   // threads read global → write shared
__syncthreads();          //   ← everyone waits for the load to finish
compute_on_tile();        // math units run; the load units now idle
__syncthreads();          //   ← everyone waits before reusing the buffer
```

Two things waste time. First, the load goes *through registers*: each thread reads a
global value into a register, then writes it to shared — the thread is busy babysitting a
copy instead of doing math. Second, **load and compute never overlap**: the barrier forces
all loading to finish before any computing starts. The memory pipe and the math pipe take
turns, and one is always idle.

## `cp.async` — copy that doesn't block the thread (Ampere)

`cp.async` issues a **global → shared copy that bypasses registers** and returns
immediately. The thread keeps going; the copy completes in the background. You group
copies into a stage, `commit` it, then later `wait` for it — so you can *issue the next
tile's load, then compute on the current tile while that load is in flight*.

```cpp
cp.async(shared[next], global[next]);   // issue: returns immediately, no register stop
cp.async.commit_group();                // bundle this stage
// ... compute on shared[cur] while the copy above streams in ...
cp.async.wait_group(0);                 // only now block, if the next tile isn't ready
__syncthreads();
```

Combine it with **double buffering** (two shared tiles, `cur`/`next`) and the K-loop
becomes a software pipeline:

<svg class="dojo-diagram" viewBox="0 0 760 150" role="img" aria-label="Without overlap, load and compute alternate. With cp.async double buffering, each tile's load overlaps the previous tile's compute.">
  <text class="mono" x="10" y="20">serial (load, then compute, then load…)</text>
  <rect class="fill-faint stroke-accent" x="20" y="30" width="90" height="24" rx="4"/><text class="mono" x="65" y="47" text-anchor="middle">load 0</text>
  <rect class="fill-accent" x="110" y="30" width="120" height="24" rx="4"/><text class="mono" x="170" y="47" text-anchor="middle" fill="#0b1500">compute 0</text>
  <rect class="fill-faint stroke-accent" x="230" y="30" width="90" height="24" rx="4"/><text class="mono" x="275" y="47" text-anchor="middle">load 1</text>
  <rect class="fill-accent" x="320" y="30" width="120" height="24" rx="4"/><text class="mono" x="380" y="47" text-anchor="middle" fill="#0b1500">compute 1</text>
  <text class="mono" x="10" y="92">pipelined (load N+1 overlaps compute N)</text>
  <rect class="fill-faint stroke-accent" x="20" y="102" width="90" height="24" rx="4"/><text class="mono" x="65" y="119" text-anchor="middle">load 0</text>
  <rect class="fill-accent" x="110" y="102" width="120" height="24" rx="4"/><text class="mono" x="170" y="119" text-anchor="middle" fill="#0b1500">compute 0</text>
  <rect class="fill-faint stroke-accent" x="110" y="128" width="90" height="18" rx="4"/><text class="mono" x="155" y="142" text-anchor="middle">load 1</text>
  <rect class="fill-accent" x="230" y="102" width="120" height="24" rx="4"/><text class="mono" x="290" y="119" text-anchor="middle" fill="#0b1500">compute 1</text>
  <rect class="fill-faint stroke-accent" x="230" y="128" width="90" height="18" rx="4"/><text class="mono" x="275" y="142" text-anchor="middle">load 2</text>
</svg>

**Predict before reading on:** if a GEMM is compute-bound on the tensor cores, what does
hiding the loads buy you — and what does it buy a *memory-bound* kernel? *(Compute-bound:
you hide load latency behind math that was going to run anyway, approaching the math-pipe
roofline. Memory-bound: little — you were already limited by bytes, and `cp.async` moves
the same bytes; it removes the register detour and the stall, not the traffic.)* This is
the Level 4 roofline question, asked inside the kernel.

To review the tiling and hierarchy this builds on, revisit the
[tiled transpose](level03-memory-hierarchy.md) and shared-memory widgets:

<div data-dojo="mem-hierarchy"></div>

## TMA — let the hardware move the tile (Hopper)

`cp.async` still spends *threads* to issue the copies — 128 lanes each firing copy
instructions. The **Tensor Memory Accelerator** replaces that with a single descriptor:
one thread describes a whole tile (a 2-D/3-D region of global memory) and the TMA engine
performs the bulk transfer asynchronously, signalling a barrier when it lands.

```cpp
// one thread issues the whole tile copy; the TMA engine does the rest
cp.async.bulk.tensor(shared_tile, tma_descriptor, coords, barrier);
// ... compute while the engine streams the tile in; then wait on the barrier ...
```

The win is *instruction overhead*: the 127 other threads never run copy instructions at
all, so they're free to compute. TMA also handles the address arithmetic and boundary
clamping for the tile in hardware. This is Hopper+ only and must be measured on that
hardware to confirm the overlap.

## Thread-block clusters & distributed shared memory (Hopper)

Until now, two blocks could only communicate through **global memory** — a block's shared
memory was private (Level 5). Hopper adds a tier *between* block and grid: a **thread-block
cluster** is a small group of blocks, co-scheduled on neighbouring SMs, that can:

- **read each other's shared memory** directly (Distributed Shared Memory, DSM), and
- **synchronize across the cluster** with `cluster.sync()`.

```cpp
namespace cg = cooperative_groups;
auto cluster = cg::this_cluster();
float* peer  = cluster.map_shared_rank(smem, otherBlockRank); // read a sibling block's smem
cluster.sync();                                               // barrier across the cluster
```

This lets a tile be *shared* across several blocks without the global-memory round-trip a
cross-block algorithm used to require — a bigger effective tile than one block's shared
memory allows, and a class of algorithms that previously needed a kernel boundary
([Level 9](level09-multi-kernel.md)) for their cross-block sync. The same
cooperative-groups API scales from a sub-warp tile, to the block, to the cluster, to the
whole grid (the persistent-worker grid barrier from
[Level 11](level11-orchestration.md)).

!!! warning "This is the frontier — earn it"
    These features have a high bar. `cp.async` pays off when a kernel is compute-bound and
    starved by load latency; TMA and clusters matter mainly inside high-performance GEMM /
    attention kernels and the libraries that ship them (CUTLASS, cuDNN). For most work, the
    right move is still *call the library* ([Level 7](level07-libraries.md)) that already
    pipelines tiles for you. Reach for these when you are *writing* that library — and
    profile on the actual Ampere/Hopper/Blackwell target, because none of it can be
    confirmed on older hardware.

## Your reps

- **Double-buffer a tiled GEMM with `cp.async`** (Ampere+): take the Level 9
  register-blocked GEMM and overlap each K-tile's load with the previous tile's compute.
  Predict the speedup from the roofline first, then measure with `ncu` — look at whether
  memory-pipe stalls on the math warps drop.
- **Read the CUTLASS pipeline**: open a CUTLASS GEMM `mainloop` and find the `cp.async`
  commit/wait stages and (on Hopper) the TMA + cluster code. You're reading the production
  version of everything on this page.
- For the hardware view of *why* these exist — async copy units, the TMA engine, SM
  occupancy under big tiles — see the [Architecture track](track-architecture.md).

??? question "Self-check"
    You add `cp.async` double buffering to a GEMM and see no speedup in `ncu`. Name two
    plausible reasons. *(1) The kernel is memory-bound, not compute-bound — there's no
    compute to hide the loads behind, so overlapping load and compute changes nothing.
    (2) The tile/buffer sizing left no real overlap: the `wait_group` blocks immediately
    because the next load wasn't issued early enough, or shared memory was too small to
    hold two tiles, dropping you back to single-buffering.)*

→ Back to the [Advanced Tracks](track-architecture.md) for the hardware and library views.
