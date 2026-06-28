# Level 8 — Synchronization Scopes & Idioms

> **The question:** *When data moves from one thread, warp, block, kernel, or
> stream to another, what is the smallest thing that must wait?*

Synchronization is not an API list. It is a **scope decision**. Pick too small a
scope and you have a race. Pick too large a scope and you serialize work that could
have overlapped.

Levels 6 and 7 gave you `__syncthreads()` and the warp masks by example. This level
lines up *every* scope on one ladder so you can pick deliberately instead of by
habit.

## The decision ladder

Run every synchronization question through one prompt:

> **Who must see whose writes before continuing?**

Answer that, and the scope picks itself. Climb only as high as the answer forces
you.

<div data-dojo="sync-scope"></div>

| Climb to… | Primitive | What waits |
|-----------|-----------|------------|
| Warp / selected lanes | `__syncwarp(mask)` | named lanes in one warp |
| Block | `__syncthreads()` | all threads in one block |
| Tile (Cooperative Groups) | `tile.sync()` | threads in a programmer-defined tile |
| Grid | `cg::this_grid().sync()` | all blocks — **cooperative launch only** |
| Kernel boundary | later same-stream work | all blocks of the previous kernel |
| Stream / event | `cudaEventRecord` + `cudaStreamWaitEvent` | one queue waits for a point in another |
| CPU | `cudaStreamSynchronize` / `cudaDeviceSynchronize` | the host thread waits |

The rule of thumb: **start at the smallest correct scope.** A warp-local register
exchange needs no block barrier. A cross-warp shared-memory handoff does. A
cross-block handoff needs a kernel boundary, a cooperative launch, or a
global-memory protocol of atomics and fences.

!!! tip "Key takeaway"
    Every primitive below is the *same question* answered at a different radius. Memorize
    the question, not the API table.

---

## Warp / selected-lane sync — `__syncwarp(mask)`

- **When to use:** lanes in *one* warp exchange through memory, or need to
  reconverge after a divergent branch.
- **Synchronizes:** the lanes named in `mask`, within one warp.
- **Does *not* synchronize:** any other warp; nothing block-wide.
- **Minimal idiom:**

```cpp
unsigned mask = __ballot_sync(0xffffffff, pred);  // name the participants first
if (pred) {
    smem[threadIdx.x] = value;
    __syncwarp(mask);                              // order writes for these lanes
    value = smem[threadIdx.x ^ 1];
}
```

- **Common mistake:** `__syncwarp(0xffffffff)` *inside* a branch only some lanes
  enter — it asks for absent lanes to participate. Derive the mask with
  `__ballot_sync` before the branch (Level 7, Example 3).

!!! warning "Why `__syncwarp()` even exists"
    Pre-Volta, a converged warp moved in lockstep and you rarely needed it. On
    Volta+, independent thread scheduling means lanes can drift after divergence, so
    warp-level memory exchange needs explicit ordering. `__syncwarp()` reconverges
    and orders memory **for one warp** — it is not a cheaper `__syncthreads()`.

## Block sync — `__syncthreads()`

- **When to use:** communication crosses warp boundaries inside one block (the
  classic shared-memory tile handoff).
- **Synchronizes:** all threads in the block — barrier *and* memory ordering for
  shared/global writes before it.
- **Does *not* synchronize:** other blocks; the host.
- **Minimal idiom:**

```cpp
smem[threadIdx.x] = load(threadIdx.x);
__syncthreads();                       // every thread must reach this
out[threadIdx.x] = smem[neighbor(threadIdx.x)];
```

- **Common mistake:** calling it inside a divergent branch so some threads never
  arrive → deadlock. *Every* thread in the block must reach the *same*
  `__syncthreads()`.

The contrast with the warp barrier is the whole point — different radius, different
guarantee:

```text
   __syncwarp(mask)                 __syncthreads()
   ┌───────────────┐                ┌───────────────────────────────┐
   │ warp 0: ✓ wait │ warp 1: free  │ warp 0 ✓ │ warp 1 ✓ │ warp 2 ✓ │
   └───────────────┘                └───────────────────────────────┘
   one warp only                    the whole block
```

## Tile sync — `tile.sync()`

- **When to use:** the algorithm's cooperation unit is a subgroup that is *not* "the
  whole block" — and you want that intent visible in the type.
- **Synchronizes:** exactly the threads in that tile.
- **Does *not* synchronize:** threads outside the tile.
- **Minimal idiom:**

```cpp
namespace cg = cooperative_groups;
auto block = cg::this_thread_block();
auto tile  = cg::tiled_partition<16>(block);
tile.sync();                           // only these 16 threads
```

- **Common mistake:** assuming a tile is always warp-local. `tile<16>` may split a
  warp, `tile<32>` usually *is* a warp, `tile<64>` **spans warps** — and only the
  ≤32 case avoids needing block-level ordering. Predict which side of 32 your tile
  lands on.

## Grid sync — cooperative launch

- **When to use:** *all* blocks must rendezvous **without** ending the kernel —
  e.g. an in-kernel multi-phase algorithm that would otherwise need a relaunch.
- **Synchronizes:** every block in the grid, at the barrier.
- **Does *not* synchronize:** anything in another kernel or stream; and it only
  works if the launch was cooperative.
- **Minimal idiom:**

```cpp
auto grid = cg::this_grid();
produce_phase(grid);
grid.sync();                           // all blocks meet here
consume_phase(grid);
```

- **Common mistake:** using it after an ordinary `<<<grid, block>>>` launch. A grid
  barrier requires `cudaLaunchCooperativeKernel` **and** a grid sized to stay
  co-resident (query `cudaOccupancyMaxActiveBlocksPerMultiprocessor`). Oversubscribe
  the SMs and it deadlocks — the late blocks can't start, the early ones won't
  finish. This is the *exception* that proves the Level 3 rule, not the default.

!!! note "Default to the kernel boundary"
    Cooperative grid sync is a specialist tool. For most cross-block barriers, the
    plain kernel boundary below is simpler, has no occupancy constraint, and is what
    CUB/Thrust use.

## Kernel-boundary barrier

- **When to use:** blocks must exchange results — the everyday cross-block barrier.
- **Synchronizes:** all blocks of the earlier kernel finish before any block of the
  next same-stream kernel starts.
- **Does *not* synchronize:** the host (launches are async); kernels in *other*
  streams unless an event links them.
- **Minimal idiom:**

```cpp
stage_a<<<grid, block, 0, stream>>>();   // all blocks finish...
stage_b<<<grid, block, 0, stream>>>();   // ...before any of these start
```

```text
   stream:  [ stage_a : all blocks ] ──── boundary = global barrier ────▶ [ stage_b ]
```

- **Common mistake:** expecting this ordering *across* streams. The boundary orders
  one stream only.

This is why reductions, scans, histograms, and many graph algorithms are written as
multiple kernels: the boundary is the cheapest correct cross-block barrier.

## Stream / event dependency

- **When to use:** order work *between* asynchronous queues without stalling the
  CPU.
- **Synchronizes:** the waiting stream waits for a recorded point in another stream.
- **Does *not* synchronize:** the host (it keeps running); and it does not share
  registers or shared memory — streams only order global memory, copies, kernels,
  events.
- **Minimal idiom:**

```cpp
kernelA<<<grid, block, 0, streamA>>>();
cudaEventRecord(done, streamA);          // mark a point in A
cudaStreamWaitEvent(streamB, done);      // B waits for that point — CPU does not
kernelB<<<grid, block, 0, streamB>>>();
```

<div data-dojo="stream-event-dependency"></div>

```text
   streamA:  [ kernelA ] ●record(done)
                          \
                           ▶ wait(done)
   streamB:  ............... [ kernelB ]   (B's queue blocks until A hits the dot)
```

- **Common mistake:** fanning in N producers but waiting on only one event — the
  consumer can start before the others finish. One event per producer, wait on all
  of them (you'll see this exact bug at Level 13).

## CPU wait

- **When to use:** the host genuinely needs the result *now* — to read it back,
  time it, or branch on it.
- **Synchronizes:** the host thread blocks until the GPU reaches the point.
- **Does *not* synchronize:** nothing the GPU couldn't already order itself — so
  using it for *device-side* ordering is wasteful.
- **Minimal idiom:**

```cpp
cudaStreamSynchronize(stream);   // or cudaEventSynchronize / cudaDeviceSynchronize
float result = host_buffer[0];   // now safe to read
```

- **Common mistake:** a stray `cudaDeviceSynchronize()` in a hot loop. It serializes
  everything and kills the overlap streams were meant to buy (the classic Level 10
  "no speedup" cause).

## Fences are not barriers

A memory fence orders the *visibility* of one thread's memory operations. It does
**not** make any other thread wait:

```cpp
__threadfence_block();   // visible to this block
__threadfence();         // visible to the whole device
__threadfence_system();  // visible to host + peers
```

- **What it does:** "my earlier writes are visible before my later ones."
- **What it does *not* do:** "everyone has arrived." A fence has no rendezvous.

A fence is a building block for a global-memory protocol *together with* atomics
(e.g. the last-block-finishes flag in a single-pass reduction). But if your bug is
that another thread/block/stream **might not have reached a point yet**, you need a
synchronization primitive or an explicit dependency — not a fence.

!!! warning "Common trap: \"I added `__threadfence()` and the race went away\""
    It may have *hidden* the race by changing timing. A fence orders one thread's
    own writes; it never proves a *different* thread already ran. Ask the ladder
    question again — you probably need a barrier or an event.

---

## Choose the primitive

For each, name the smallest correct primitive *before* peeking.

??? question "1. 32 lanes of one warp sum their values through registers."
    None beyond the `_sync` mask. Register shuffles inside a warp carry their own
    ordering via the participant mask — no barrier needed.

??? question "2. Warp 0 fills a `__shared__` tile that warp 1 then reads."
    `__syncthreads()`. The handoff crosses warp boundaries inside the block;
    `__syncwarp()` would only order warp 0.

??? question "3. Block 0 produces partial sums that block 7 must combine."
    A kernel boundary (or a cooperative grid sync). Blocks aren't co-resident or
    ordered within one ordinary launch.

??? question "4. Stream B's kernel must start only after stream A's copy lands."
    `cudaEventRecord` on A + `cudaStreamWaitEvent` on B. No CPU stall needed.

??? question "5. The host must print a reduced scalar the GPU just computed."
    `cudaStreamSynchronize` (or `cudaEventSynchronize`), *then* read host memory.

??? question "6. A single-pass reduction needs the last finishing block to know it's last."
    Atomics **plus** `__threadfence()` — the fence makes the partial writes visible
    before the atomic flag flips. A fence alone wouldn't make blocks wait; the atomic
    is what elects the last block.

## Your reps

- Fix the cross-warp shared-memory race from Level 7 by choosing the smallest
  correct scope, and justify why the next-smaller scope fails.
- Given a predicate branch, write the mask once with `__ballot_sync` and say which
  lanes may call the matching `_sync` intrinsic.
- Partition a block into `tile<16>`, `tile<32>`, `tile<64>` and predict which are
  warp-local and which cross warp boundaries.
- Wire two streams with an event dependency and read the `nsys` timeline: the host
  should launch and continue; only the dependent stream waits.

??? question "Self-check"
    Why is `__threadfence()` not a replacement for `__syncthreads()`? *(A fence
    orders memory visibility for the calling thread. It does not wait for the other
    threads to arrive, so it cannot prove that another thread has already written the
    value you plan to read.)*

→ Continue to [Level 9 — CUDA Libraries](level09-libraries.md). Levels 10, 11, and
13 all lean back on this ladder — streams and events are just synchronization scopes
that live *outside* the kernel.
