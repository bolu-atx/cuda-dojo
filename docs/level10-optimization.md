# Level 10 — Production Optimization

> **The question:** *Your kernel takes 20 ms. The customer needs 1 ms. You can't
> get there by guessing. What does the hardware actually tell you when you ask it?*

This is where NVIDIA engineers spend their careers. The discipline: **profile →
find the dominant limiter → fix exactly that → re-profile.** No speculative
optimization. The tools don't lie; your intuition does.

## The two tools, and what each answers

| Tool | Scope | Answers |
|------|-------|---------|
| **Nsight Systems** (`nsys`) | whole timeline | Where does wall-clock go? Am I copy-bound? Are kernels overlapping? Gaps? |
| **Nsight Compute** (`ncu`) | one kernel, deeply | *Why* is this kernel slow? Memory vs compute bound, occupancy, stalls, the roofline dot. |

Always start with `nsys` — it stops you from optimizing a kernel that's only 5% of
runtime while a `cudaMemcpy` eats 60%. *Then* `ncu` the kernel that actually
dominates.

```bash
nsys profile -o timeline ./demo            # then open in the Nsight Systems GUI
ncu --set full -k my_kernel ./demo         # full metric set for one kernel
```

## The optimization decision tree

Re-derive *bound by what?* first (Level 4) — the rest follows:

<div data-dojo="roofline"></div>

**Memory-bound?** (most imaging kernels)

- Achieved vs peak bandwidth — `ncu` reports it directly.
- Coalescing: check "global load/store efficiency". Below 100%? Fix your access
  pattern (Level 2/4).
- Shared-memory bank conflicts metric → pad your tiles (Level 5).
- L2 hit rate low? Restructure for reuse / tiling (Level 3).

<div data-dojo="coalescing"></div>

**Compute-bound?**

- Instruction mix — are you doing expensive ops (`div`, `sin`) you could replace?
- Use intrinsics (`__fdividef`, `--use_fast_math`) where precision allows.
- Tensor cores for matmul-shaped work (Level 11).

**Latency-bound / low occupancy?** (ALUs idle, neither wall hit)

- **Register pressure** — too many registers/thread caps resident warps. Check
  `ncu`'s "registers per thread"; cap with `__launch_bounds__(maxThreads)` or
  `-maxrregcount`.
- **Shared memory per block** too high → fewer blocks per SM.
- Increase parallelism: more blocks, or more work per thread (ILP) so each warp
  has independent instructions to hide latency.

## The knobs, ranked by payoff

1. **Coalescing & access pattern** — usually the biggest single win.
2. **Occupancy** (registers/shared mem) — enables latency hiding.
3. **Shared-memory tiling / reuse** — cut DRAM traffic.
4. **Loop unrolling / ILP** (`#pragma unroll`) — more independent work per warp.
5. **Fast-math intrinsics** — when precision budget allows.
6. **Launch config** — block size that maximizes occupancy for *this* kernel.

!!! tip "The 20 → 1 ms mindset"
    Big speedups are *layers*, not one trick. Typical arc: 20 ms (uncoalesced) →
    5 ms (fix coalescing) → 2 ms (tile in shared memory) → 1 ms (raise occupancy +
    unroll). After each change, re-profile — the dominant limiter *moves*, and so
    must your attention.

## Your reps

- Take one Level 4 filter and drive it down in measured stages. Keep a table:
  *version → time → bound → what changed*. That table is the deliverable.
- Open the `nsys` timeline and the `ncu` roofline for your kernel and read the
  dot's position. Make a prediction, then verify it against the metric.

??? question "Self-check"
    `ncu` shows 95% achieved bandwidth, 35% occupancy, and the roofline dot sitting
    on the memory ceiling. Should you chase higher occupancy? *(No. You're already
    at 95% of the bandwidth ceiling and memory-bound — occupancy is "good enough"
    because there's enough latency hiding to saturate the bus. The only way faster
    is to move fewer bytes: more reuse/fusion to raise arithmetic intensity.)*

→ Continue to [Level 11 — Advanced CUDA](level11-advanced.md)
