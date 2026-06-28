# Level 7 — Warp Programming

> **The question:** *Your block reduction uses shared memory and three
> `__syncthreads()`. But the last 32 threads are one warp. Can they exchange
> register values directly — and what does the mask in `__shfl_down_sync(mask, …)`
> actually promise?*

Yes, they can. This is where you stop treating threads as always-independent and
start programming the hardware execution group directly. Warp intrinsics let the 32
lanes of a warp **read each other's registers** — no shared memory, no block
barrier.

Level 3 told you a block is sliced into warps of 32 consecutive threads. Level 6
made the *block* a team. This level makes the *warp* a team — a smaller, faster,
register-speed one.

## The warp is the real execution unit

Recall the SIMT widget — 32 lanes, one instruction stream:

<div data-dojo="simt"></div>

**Intuition.** Think of a warp as a 32-wide SIMD register where each lane is also
an addressable thread. Because the lanes already share an issue slot, moving a
value from lane 6 to lane 5 is a register-file read, not a memory round-trip.

**The precise rule (and the Volta+ nuance).** An older shortcut said a warp is *32
lanes in perfect lockstep*, and people leaned on that lockstep as if it were a
*synchronization guarantee*. It is not. On Volta and later, **independent thread
scheduling** means lanes can make independent progress after a branch. Lockstep is
still a good model for *throughput*; it is **not** a safe model for *correctness*.

That is the entire reason the modern intrinsics are **`_sync`-suffixed** and take a
**mask**:

> A mask is just the set of lanes participating in this one warp operation — one
> bit per lane, bit *i* set means "lane *i* is in."

| Intrinsic | What it does |
|-----------|--------------|
| `__shfl_sync(mask, val, srcLane)` | read `val` from another lane's register |
| `__shfl_down_sync(mask, val, d)` | read from lane `id + d` (the reduction workhorse) |
| `__shfl_xor_sync(mask, val, m)` | butterfly exchange — full reductions / scans |
| `__ballot_sync(mask, pred)` | 32-bit mask of which participating lanes have `pred` true |
| `__any_sync` / `__all_sync` | warp-wide OR / AND of a predicate |
| `__activemask()` | which lanes are *currently* converged and active |

!!! warning "Common trap: \"lockstep means I don't need a mask\""
    On pre-Volta hardware you often got away with maskless `__shfl`. On Volta+ a
    shuffle that reads from a lane *not in the mask* (or not currently executing)
    returns **garbage**, silently. The mask is not boilerplate — it is the
    correctness contract for the operation.

<div data-dojo="warp-lanes"></div>

!!! tip "Key takeaway"
    A warp is the real execution group, but on modern GPUs lockstep is a
    *scheduling tendency*, not a synchronization guarantee. **Correctness comes from
    the mask, not from lockstep.**

## Masks: the set of participating lanes

A mask answers one question: *which lanes are taking part in this operation right
now?* Every lane named in the mask must execute the **same** `_sync` call.

<div data-dojo="sync-mask"></div>

- All 32 lanes converged → `0xffffffff`.
- Only some lanes (a predicated branch) → compute the mask from the predicate.

The cardinal rule: **derive the mask before the branch, with `__ballot_sync`**, so
every lane agrees on who's in.

```cpp
unsigned mask = __ballot_sync(0xffffffff, active);  // every lane votes here
// ...now mask names exactly the lanes with `active == true`
```

!!! tip "Key takeaway"
    Masks define participants. A `_sync` intrinsic is a contract among exactly the
    lanes in its mask — no more, no fewer.

## Example 1 — full-warp reduction (shared memory, deleted)

When all 32 lanes participate, the mask is `0xffffffff`. A full-warp sum in five
shuffles — no `__shared__`, no `__syncthreads()`:

```cpp
__device__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;   // lane 0 holds the warp's total
}
```

It's the same halving tree from Level 1 — but the data moves through the **register
file**, not shared memory:

<div data-dojo="reduction"></div>

**Predict before you read on:** how many `__syncthreads()` does this loop need?
*(Zero — the exchange never leaves the warp, so there is no cross-warp memory to
order. The mask does the work a barrier would otherwise do.)*

The production reduction stacks two layers: **warp-reduce within each warp → write
one partial per warp to shared memory → warp-reduce those partials.** One
`__syncthreads()` total instead of five. This is how CUB and Thrust do it under the
hood.

!!! tip "Key takeaway"
    Shuffles move *register values* between lanes. No shared memory, no block
    barrier — the data never leaves the warp's register file.

## Example 2 — predicated tail reduction

The last block of a reduction often has a partial warp: some lanes have real data,
some are past the end. Compute the participant mask **before** the branch, then feed
it to every shuffle:

```cpp
unsigned mask = __ballot_sync(0xffffffff, active);  // who has real data
float v = active ? x : 0.0f;

for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_down_sync(mask, v, offset);         // same mask each step
```

Every lane named in `mask` runs the same `_sync` call. The mask is the contract —
inactive lanes are simply absent from it.

## Example 3 — the wrong full-mask-in-branch pattern

This compiles and *looks* fine. It is a bug:

```cpp
if (active) {                                        // only some lanes enter
    v += __shfl_down_sync(0xffffffff, v, 16);        // ❌ mask claims all 32
}
```

`0xffffffff` promises all 32 lanes participate, but the inactive lanes never
entered the branch to make the call. The shuffle reads from lanes that aren't
there → undefined values, no error. **Name only the lanes that are actually
executing** (Example 2's `mask`).

!!! warning "Common trap: \"`0xffffffff` is always safe\""
    A full mask is only correct when the whole warp is converged at that line. The
    moment a predicate splits the warp, the full mask is a lie. Reach for
    `__ballot_sync` instead.

## Example 4 — warp-local memory exchange with `__syncwarp()`

Shuffles move *registers*. When lanes instead exchange through *memory* (shared or
global) **within one warp**, you need `__syncwarp(mask)` to order those accesses on
Volta+ — because the lanes may not be marching in step:

```cpp
smem[lane] = value;          // each lane writes its slot
__syncwarp();                // order writes before reads, within this warp
out[lane] = smem[(lane + 1) & 31];   // now safe to read a neighbor's slot
```

`__syncwarp()` reconverges and orders memory **only for lanes in this warp**. It is
*not* a smaller `__syncthreads()`.

## Cross-warp is a different scope

Here is the misconception that bites hardest. `__syncwarp()` in warp 0 does
**nothing** for warp 1:

<div data-dojo="cross-warp-race"></div>

```cpp
if (warp == 0) { smem[lane] = produce(lane); __syncwarp(); }  // warp 0 only
if (warp == 1) { consume(smem[lane]); }                        // ❌ may run first
```

Warp 1 can read `smem` before warp 0 has written it — `__syncwarp()` never made
warp 1 wait. **Communication that crosses warp boundaries needs `__syncthreads()`,
and every thread in the block must reach it.** That's the Level 8 scope ladder in
miniature.

!!! tip "Key takeaway"
    Warp-local is **not** block-wide. A warp intrinsic or `__syncwarp()` coordinates
    one warp; the moment data crosses warps, climb to `__syncthreads()`.

## Warp aggregation: cheap atomics

Hammering a global counter with 32 `atomicAdd`s per warp serializes them. Instead,
the warp **votes and aggregates**: `__ballot_sync` counts how many lanes want to
increment, lane 0 does *one* `atomicAdd` of the total, then `__shfl_sync`
broadcasts the base offset back. 32 atomics → 1. This is the core trick behind fast
histograms and stream compaction.

## Your reps

| Project | Intrinsic focus |
|---------|-----------------|
| **warp reduction** (worked) | `__shfl_down_sync` — then benchmark vs the shared-mem version |
| **warp histogram** (your turn) | `__ballot_sync` + warp-aggregated `atomicAdd` |

Stretch rep: a **warp prefix sum** with `__shfl_up_sync` — an inclusive scan in
log₂32 = 5 steps.

??? question "Self-check"
    Why does a warp shuffle need no `__syncthreads()` but a block-wide reduction
    does? *(A shuffle moves data among lanes in one warp through registers, and the
    `_sync` mask names the participating lanes. A block-wide reduction crosses warp
    boundaries through shared memory, so the block needs a block-wide barrier before
    another warp reads those partials.)*

→ Continue to [Level 8 — Synchronization Scopes & Idioms](level08-synchronization-scopes.md)
— you just used three different scopes; next you'll see the whole ladder.
