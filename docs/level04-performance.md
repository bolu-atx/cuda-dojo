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

## The exercises — `levels/level04_performance/`

This level's deliverable is **a number, not a kernel**. The demo does the
instrumenting for you: for each kernel it computes arithmetic intensity *before*
timing, then prints achieved GB/s, GFLOP/s, and **% of theoretical peak** with a
memory- vs compute-bound verdict. Your job is to read those numbers, classify the
kernel, then earn back the gap.

| File | Status | The payoff |
|------|--------|------------|
| `box_blur_naive.cu` | ✅ **baseline** | A radius-`r` box blur; each thread re-fetches its whole `(2r+1)²` window from global. Compute its AI by hand first, then confirm the demo agrees it's memory-bound. |
| `box_blur_opt.cu` | 📝 **your turn** | Same result, but move far fewer bytes — shared-tile + halo, or exploit separability (`(2r+1)²` reads → `2(2r+1)`). Target: **≥80% of peak BW**. Name which of the four sins you removed. |
| `sobel.cu` | ✅ **complete** | A second memory-bound stencil. Use it to practice the habit: predict AI ≈ 0.5 FLOP/B and notice the `sqrt` does **not** flip the classification. |

What to get out of it:

- **Compute AI first, every time.** Before touching the optimized kernel, predict
  its achieved bandwidth and which wall governs. Someone who understands lands
  within ~2×; cargo-culting can't. The demo is your answer key.
- **Know when to stop.** A memory-bound kernel at ≥80% of peak is essentially
  done — going faster means *raising arithmetic intensity* (moving fewer bytes),
  not chasing the last 10% of the bandwidth ceiling.

!!! tip "Prove it, don't trust it"
    `make level04-test` pins correctness (the optimized blur must match the
    baseline to tolerance, at awkward sizes). For the performance claims, the demo
    self-reports GB/s, but the **source of truth** is the profiler — the demo even
    prints the commands:

    ```bash
    nsys profile  ./build/levels/level04_performance/level04_demo
    ncu --set full ./build/levels/level04_performance/level04_demo
    ```

    Note: the demo's GB/s counts *minimum* traffic (one read + one write). The
    naive kernels move more because of overlapping re-fetches — that excess is
    exactly the headroom `ncu` will show you, and exactly what your optimized
    kernel removes.

??? question "Self-check"
    A kernel achieves 1400 GB/s on a GPU with 1555 GB/s peak and AI=0.5. Is there
    meaningful headroom? *(No — at AI=0.5 you're far left of the ridge, so the
    bandwidth ceiling governs, and you're at 90% of it. To go faster you must
    *raise arithmetic intensity* (reuse data so you move fewer bytes), not chase
    the remaining 10%.)*

→ Continue to [Level 5 — Shared-Memory Algorithms](level05-shared-memory.md)
