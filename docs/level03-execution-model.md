# Level 3 — Logical vs Physical Execution

> **The question:** *When you launch a grid of blocks, where do those blocks
> actually run, and why does that decide who can cooperate?*

You just spent Level 2 mapping threads onto data. Now learn **where those threads
run** — because that, not your indexing, decides which threads can talk to each
other.

CUDA hands you two machines at once:

- the **logical programming model** you *write* — grid, block, thread;
- the **physical execution model** the GPU *runs* — SM, warp, lane.

Almost every CUDA misconception is a leak between these two machines: expecting a
logical guarantee (the whole grid) from a physical resource that can't provide it
(one SM at a time). Keep them separate and the rest of the dojo falls into place.

<div data-dojo="execution-model"></div>

## The two machines, side by side

You never name the right column in a kernel launch. The runtime maps the left
column onto it.

```text
   LOGICAL (you write)            PHYSICAL (hardware runs)
   --------------------           ------------------------
   Grid                           GPU
   └── Block          ── runtime ── SM (streaming multiprocessor)
       └── Thread       schedules     └── Warp (32 lanes, lockstep issue)
                                          └── Lane (one thread's slot)
```

**Intuition (from the CPU you know).** A thread pool is the logical unit you
submit; cores are the physical unit that runs it. You don't pin tasks to core 3.
CUDA is the same idea with one extra twist: the *block* — not the thread — is the
unit the scheduler places, and it places it **whole**.

**The precise rule:** the programmer owns the grid/block/thread hierarchy; the
runtime owns the SM/warp/lane hierarchy. An SM is a hardware resource — schedulers,
register file, execution pipes, shared memory — not something you address by name.

### What you control / what CUDA controls

| You control | CUDA controls |
|-------------|---------------|
| `gridDim`, `blockDim` (how many blocks/threads) | which SM runs which block |
| the thread → data mapping (`blockIdx`, `threadIdx`) | when each block starts and retires |
| how much shared memory / how many registers a block asks for | how many blocks are resident on an SM at once |
| where you place barriers and which scope they use | how the 32 lanes of a warp are issued |
| the *logical* shape of the work | the *physical* schedule that runs it |

!!! warning "Common trap: \"my block runs on SM 0\""
    You cannot choose, query reliably, or depend on which SM runs a block. Two
    runs of the same launch may place blocks differently. Any correctness argument
    that assumes a particular block→SM assignment is already wrong.

!!! tip "Key takeaway"
    Two machines. You write the left one; the runtime schedules it onto the right
    one. **The block is the bridge: it is the largest logical group the hardware
    guarantees runs together on one SM.**

## Why the block is the cooperation boundary

**Intuition.** Cores that share an L1 cache can hand each other data cheaply;
cores on different sockets cannot. On a GPU, the on-SM scratchpad (shared memory)
plays the role of that shared L1 — and only threads scheduled onto the *same SM at
the same time* can reach it.

**The precise rule:** a block is scheduled **wholly onto one SM** and never spans
two. So a block — and nothing larger — can rely on:

```cpp
__shared__ float tile[32][33];  // one allocation, private to this block
__syncthreads();                // waits for THIS block's threads only
```

```text
        block (whole) ── scheduled onto ──▶ one SM
        ┌───────────────────────────┐      ┌──────────────────────────┐
        │ thread thread thread ...   │      │ shared mem │ regs │ warps │
        └───────────────────────────┘      └──────────────────────────┘
                 share __shared__ + __syncthreads()
```

<div data-dojo="block-to-sm"></div>

Shared memory is **per block**. A different block gets a *different* allocation —
even if it later runs on the same SM. `__syncthreads()` is **per block** too: it
waits for the threads in one block, because the other blocks may be running on
other SMs or may not have started yet.

This is exactly why there is **no ordinary `__syncblocks()`**. The logical grid is
bigger than what the hardware guarantees is resident at once, so a "wait for every
block" barrier inside one kernel can deadlock — late blocks can't start until early
ones retire to free the SM. The correct cross-block barrier is a **kernel
boundary**:

```cpp
stage_a<<<grid, block>>>(tmp);   // all blocks finish...
stage_b<<<grid, block>>>(tmp);   // ...before any of these start (same stream)
```

??? question "Before you continue"
    Two blocks both write `tile` and you want block 1 to read what block 0 wrote.
    Which of these works, and why? (a) `__syncthreads()`, (b) a second kernel,
    (c) `__syncwarp()`. *(b. Shared memory is per-block, so block 1 can't even see
    block 0's `tile`; only a kernel boundary — or a global-memory protocol — orders
    writes across blocks. (a) and (c) are sub-block scopes.)*

### Can these threads cooperate directly?

"Directly" = through shared memory or a warp intrinsic, inside one kernel launch,
with no extra round-trip to global memory.

| Two threads are… | Cooperate directly? | Through what | Why |
|------------------|---------------------|--------------|-----|
| same warp | yes | registers via `__shfl_*_sync` | lanes share an issue slot (Level 7) |
| same block, different warp | yes | `__shared__` + `__syncthreads()` | one SM, one shared allocation (Level 6) |
| different block, same kernel | no | global memory + **kernel boundary** | blocks aren't co-resident or ordered |
| different kernel, same stream | yes, in order | global memory; boundary orders them | the launch order is the barrier (Level 8) |
| different stream | only via events | `cudaStreamWaitEvent` | streams are independent queues (Level 10) |

!!! tip "Key takeaway"
    The cooperation boundary is **physical, not logical**. Same SM → cheap
    cooperation. Crossing SMs → you must climb to a coarser scope (kernel boundary,
    cooperative launch, or a global-memory protocol).

## Warps: where the two machines meet

**Intuition.** You wrote scalar code for one thread. The SM doesn't issue one
thread at a time — it issues **32 lanes of one warp** together, like a 32-wide
SIMD unit whose lanes happen to be addressable as threads.

**The precise rule:** a block is sliced into warps of 32 consecutive threads. For
a 1-D block the IDs are pure bit math:

```cpp
int lane = threadIdx.x & 31;  // position within the warp  (threadIdx.x % 32)
int warp = threadIdx.x >> 5;  // which warp                (threadIdx.x / 32)
```

```text
blockDim.x = 256  →  8 warps
 threadIdx.x:  0 .. 31 | 32 .. 63 | 64 .. 95 | ... | 224 .. 255
 warp:            0     |    1     |    2     | ... |    7
 lane:         0 .. 31  | 0 .. 31  | 0 .. 31  | ... | 0 .. 31
```

So `threadIdx.x == 70` is warp 2, lane 6. It's still an ordinary thread; the
warp/lane view just names how the hardware groups it.

<div data-dojo="warp-lanes"></div>

This single fact pays off everywhere downstream:

| Warp fact | Where it returns |
|-----------|------------------|
| consecutive lanes hold consecutive `threadIdx.x` | **coalescing** — map `threadIdx.x` to the contiguous axis (Level 4) |
| a branch that splits a warp runs both sides | **warp divergence** — keep branches warp-aligned (Level 7) |
| lanes can exchange registers with no barrier | **warp intrinsics** — `__shfl_*_sync` (Level 7) |
| 32 lanes share an issue slot, blocks don't | why `__syncthreads()` is needed across warps but not within one (Level 8) |
| more resident warps hide more latency | **occupancy** — register/shared-mem budget per block (Level 5) |

!!! warning "Common trap: \"a warp is always in perfect lockstep\""
    A useful first picture for *performance*, but **not** a safe model for
    *correctness* on Volta and later. After a divergent branch, lanes may make
    independent progress, which is why warp intrinsics now take an explicit
    participant **mask** (Level 7). Lockstep is a scheduling tendency, not a
    synchronization guarantee.

!!! tip "Key takeaway"
    You don't launch warps, but you program *for* them. The warp is the lens that
    turns Level 2's mapping choices into Level 4–8's performance and correctness
    rules.

## Scope is the whole story

Every CUDA coordination primitive has a **scope**. Before reaching for an API, ask
the one question that runs through the rest of the dojo: **who must wait for whom?**

| Scope | What can coordinate | First appears |
|-------|---------------------|---------------|
| Lane / warp | selected lanes in one warp | Level 7 |
| Block | threads in one block | Level 6 |
| Grid | all blocks — only with a cooperative launch | Level 8 |
| Kernel boundary | all of one kernel before later same-stream work | Level 8 / 11 |
| Stream / event | queued ops across asynchronous streams | Level 10 |
| CPU wait | the host waits on GPU progress | Level 10 |

That ladder is the spine of Level 8 and the reason the later levels are ordered the
way they are: shared memory, warp programming, streams, multi-kernel algorithms,
and tile pipelines are all just *consequences of choosing the right scope*.

## Your reps

The repo has a small **no-kernel** exercise at `levels/level03_execution_model/`
— pure arithmetic so you can drill the model without a GPU. Run
`make level03-test` to check warp/lane math and `level03_demo` to print the
mapping table.

Progressive reps, easy → hard:

1. **Compute IDs.** For `blockDim.x = 128`, name the warp and lane for
   `threadIdx.x` = 0, 31, 32, 63, 64, 127. (Check against the demo table.)
2. **Classify the scope.** For each, name the smallest scope that makes it correct:
   warp-mate exchange, cross-warp tile handoff, cross-block partial sums,
   producer/consumer across two kernels.
3. **Predict legality.** In *one* kernel, is each path legal? thread↔thread in one
   block via shared memory; block↔block via shared memory; warp↔warp via
   `__syncwarp()`; kernel→later kernel via global memory.
4. **Explain the impossibility.** Given `gridDim.x = 120` on a 10-SM GPU, explain
   why an ordinary "wait for all 120 blocks" barrier inside the kernel can deadlock
   — tie it to which facts the programmer controls vs. the runtime.
5. **Find the sync point.** For a tiled transpose, point to the exact line where a
   thread reads data another thread wrote. *That* is where the synchronization
   scope becomes non-negotiable — and Level 6 is about getting it right.

??? question "Self-check"
    Why can all threads in one block share `__shared__` memory, but two arbitrary
    blocks cannot? *(A block is scheduled wholly onto one SM and receives one
    block-local shared-memory allocation. Different blocks have different
    allocations and may run on different SMs, or at different times, so the hardware
    can't guarantee they're ever co-resident.)*

→ Continue to [Level 4 — The Memory Hierarchy](level04-memory-hierarchy.md) —
now that you know *where* threads run, *where data lives* is the next lever.
