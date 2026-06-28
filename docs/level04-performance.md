# Level 4 — Performance Thinking

> **The question:** *Your kernel takes 3 ms. Is that good? You cannot answer that
> until you know which wall you're hitting — and there are only two walls.*

Now you think like the hardware. The mindset shift: stop asking "how do I make
this faster?" and start asking **"what is the hardware limit for this kernel, and
how close am I?"**

## The only question that matters: bound by what?

Every kernel is limited by either **memory bandwidth** or **compute throughput**.
The roofline model makes this a single picture. The diagonal is the bandwidth
ceiling; the flat top is the compute ceiling; where they meet is the **ridge
point**. Your kernel's **arithmetic intensity** (FLOPs per byte moved) decides
which ceiling you're under:

<div data-dojo="roofline"></div>

- **Left of the ridge → memory-bound.** Most image processing lives here. The math
  units are starving. Optimizing the math is pointless; feed it faster.
- **Right of the ridge → compute-bound.** GEMM, transcendentals. Now reducing
  instructions and using faster math (or tensor cores) pays.

**Always compute arithmetic intensity first.** A 3×3 blur reads 9 bytes and does
~9 FLOPs per output → AI < 1 → deeply memory-bound. No amount of math cleverness
will help; coalescing and reuse will.

## The four memory sins (left-of-ridge fixes)

### 1. Uncoalesced access
The big one. A warp wants its 32 loads in one 128-byte transaction. Strided or
misaligned access multiplies your transactions and wastes the bus:

<div data-dojo="coalescing"></div>

### 2. Shared-memory bank conflicts
Shared memory has 32 banks. If 32 threads hit 32 *different* banks → full speed.
If several hit the *same* bank (e.g., a column of a 32-wide tile) → serialized.
The classic fix is **padding**: declare `__shared__ float tile[32][33]` so columns
land in different banks.

### 3. Warp divergence
Revisit the SIMT widget from Level 0 with performance eyes: a divergent branch
runs both sides at reduced width. Restructure so a branch is uniform *within a
warp*, or make both sides cheap.

### 4. Low occupancy
Too few resident warps ⇒ not enough latency to hide behind ⇒ ALUs stall. Caused
by too many registers per thread or too much shared memory per block. The
occupancy calculator (and Nsight) tells you the limiter.

The deeper scheduler question is not "is occupancy high?" but "are there enough
**eligible warps** to fill issue slots while other warps wait?" That is the bridge
from this level into [Architecture Deep Dive](track-architecture.md).

## Measure, don't guess

This is the level where you stop reasoning and start instrumenting:

```bash
nsys profile ./demo      # timeline: are you even compute-limited, or copy-bound?
ncu --set full ./demo    # per-kernel: achieved bandwidth, occupancy, the roofline dot
```

Compute **achieved GB/s** yourself too (`bytes / time`, the repo's `gbps()` helper)
and divide by your GPU's peak. Memory-bound kernel at 85% of peak bandwidth? Ship
it. At 30%? You have a coalescing or occupancy bug.

## Your reps

Take the repo's image filters and optimize them with numbers, not vibes:

- **Box blur / Sobel / morphology** — all memory-bound. Target: ≥80% of peak BW.
- For each: report **GB/s and GFLOP/s before and after**, and state which wall you
  were hitting and which sin you fixed.

??? question "Self-check"
    A kernel achieves 1400 GB/s on a GPU with 1555 GB/s peak and AI=0.5. Is there
    meaningful headroom? *(No — at AI=0.5 you're far left of the ridge, so the
    bandwidth ceiling governs, and you're at 90% of it. To go faster you must
    *raise arithmetic intensity* (reuse data so you move fewer bytes), not chase
    the remaining 10%.)*

→ Continue to [Level 5 — Shared-Memory Algorithms](level05-shared-memory.md)
