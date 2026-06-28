# Level 0 — The GPU Mental Model

> **The question to answer by the end:** *Why can a GPU run 20,000 threads at
> once when your 16-core CPU struggles past 32?*

If you try to think about a GPU like a CPU, every design decision will look
insane. So let's rebuild your intuition from scratch.

## Start with what you know: the CPU

A modern CPU core is a genius optimized for **latency** — finishing *one* stream
of instructions as fast as possible. To do that it spends most of its transistor
budget *not* on math, but on **making one thread fast**:

- huge out-of-order execution windows,
- branch predictors,
- deep cache hierarchies to avoid waiting on memory,
- speculative execution.

A CPU is a sports car: it gets *one* passenger to the destination with minimum
time. You have a handful of these cores.

## The GPU made the opposite bet

A GPU is a fleet of buses. Each "lane" is individually slow and dumb — no fancy
out-of-order machinery. But there are *thousands* of them, and the chip spends
its transistors on **arithmetic units and memory bandwidth** instead of
latency-hiding cleverness.

The GPU's bet: **if I have enough independent work, I never need to make any
single thread fast. When one thread stalls on memory, I just run another.**

That's the whole game. Latency is hidden not by caches and prediction, but by
*oversubscription* — having far more threads ready than you have execution slots.

!!! note "SIMT vs SIMD — the one distinction that matters here"
    You know SIMD: *one* instruction, one program counter, operating on a wide
    vector register (`_mm256_add_ps`). The compiler/you must explicitly pack
    lanes.

    CUDA is **SIMT** — Single Instruction, Multiple *Threads*. You write scalar
    code for *one* thread. The hardware groups 32 threads into a **warp** and
    runs them in lockstep on a SIMD unit. It looks like 32 independent threads;
    physically it's one 32-wide vector instruction. The win: divergence and
    addressing are handled by hardware, not by you hand-packing lanes.

The widget below *is* a warp. All 32 lanes share one instruction stream — until a
branch splits them.

<div data-dojo="simt"></div>

When the branch is uniform (everyone takes the same side), the warp runs at full
width. When it diverges, the hardware runs **both** sides, masking off the idle
lanes each pass. That wasted work is the SIMT tax — and the reason "GPUs like
regular work."

## The hardware, just enough of it

- A GPU has a handful of **SMs** (Streaming Multiprocessors) — think of each as a
  many-lane vector core with its own register file and scratchpad.
- Each SM runs many **warps** concurrently and switches between them *for free*
  every cycle. This is how it hides memory latency.
- **Occupancy** = how many warps are resident on an SM vs the max. More resident
  warps ⇒ more latency to hide behind. (High level for now; we quantify it at
  Level 5.)

So: thousands of threads aren't a parallelism *goal*, they're the *mechanism*. You
need a big pile of ready warps so the SM always has something to run while others
wait on the ~500-cycle trip to DRAM.

<div data-dojo="mem-latency"></div>

## Bandwidth vs FLOPS: which wall do you hit?

Two numbers describe a GPU:

- **Peak FLOPS** — how fast it can do math.
- **Memory bandwidth (GB/s)** — how fast it can feed that math from DRAM.

Their ratio is the **ridge point**. Most real kernels — and *especially* image
processing, where you touch every pixel once or twice — are **memory-bound**: the
math units sit idle waiting for bytes. Internalize this now; it reframes every
optimization you'll do.

<div data-dojo="roofline"></div>

??? question "So why can a GPU have 20,000 threads?"
    Because threads are cheap (no per-thread out-of-order hardware) and *necessary*:
    the only way to hide 500-cycle memory latency without big caches is to have a
    deep backlog of independent warps to switch to. The CPU spends silicon to make
    1 thread fast; the GPU spends it on many ALUs + bandwidth and hides latency
    with sheer thread count. Different bet, different machine.

## Your reps (code repo: `levels/level01_vector_add`)

These are deliberately trivial *as algorithms* — the point is to feel the model:

- **Vector add** — embarrassingly parallel, pure bandwidth.
- **SAXPY** — same, with one fused multiply-add.
- **Reduction** — your first taste of threads *cooperating*.

Run the demo and look at the reported **GB/s**, then compare it to your GPU's peak
(`nvidia-smi -q` or the device-info printout). How close did pure vector add get?
That gap is the rest of this dojo.

→ Continue to [Level 1 — CUDA Basics](level01-basics.md)
