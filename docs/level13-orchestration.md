# Level 13 — Orchestration Patterns

> **The question:** *You can write a fast kernel. But a real application runs dozens
> of them, plus copies, every frame — some independent, some dependent. Once the
> individual kernels are fast, what governs the program's speed?*

Up to here, the unit of thought has been *one kernel*: threads, warps, shared memory.
Past Level 10 the unit changes. The kernels are already fast; what's left is **how you
schedule them relative to each other and to the copies** — which can run at the same
time, which must wait, and how to stop the CPU from forcing everything into a line.

That skill is *orchestration*, and it has a small, reusable vocabulary. You already met
the primitives — the [scope ladder](level08-synchronization-scopes.md) (Level 8),
streams and events ([Level 10](level10-streams.md)), and the launch-overhead /
global-sync costs of composing kernels ([Level 11](level11-multi-kernel.md)). The
patterns here are just the *top rungs* of that ladder — stream/event dependencies and
CPU waits — applied at whole-program scale. This level is the **synthesis**: the named
shapes you build from those primitives. It teaches no new API; it teaches the shapes.

## CUDA is layered — the patterns live in the middle

Everything you've learned stacks. The bottom layers are *inside* a kernel; the top
layers are *between* kernels and *between* GPUs. The orchestration patterns live at
layers 4–6 — they are how independent work is expressed and made to overlap.

| Layer | You think about | Primitive |
|------:|-----------------|-----------|
| 1 | threads | `threadIdx` |
| 2 | blocks | shared memory, `__syncthreads()` |
| 3 | grid | a kernel launch |
| **4** | **streams** | **asynchronous queues** |
| **5** | **events** | **cross-stream dependencies** |
| **6** | **graphs** | **whole pipelines, replayed** |
| 7 | multi-GPU | NCCL, peer copies *(see [Architecture track](track-architecture.md))* |

The mental shift: at the kernel level you optimize *cycles*; here you optimize *who
waits for whom*. The CPU should issue work and walk away — only the **streams**
synchronize, never the host.

## Three scales of concurrency

Hold these three apart — they're the map for this level and the next:

1. **Across streams** — pipeline independent *stages* of work (this page).
2. **Within a kernel** — pipeline *tiles* through the memory hierarchy
   ([Level 16](level16-tile-pipeline.md)).
3. **Across the application** — capture the whole dependency graph and replay it
   (CUDA Graphs, below; assembled into a system at [Level 14](level14-architecture.md)).

Everything that follows is one of these. Before each pattern, **predict**: which
operations can run at the same time, and what is the one dependency that forces a wait?

## Serial → parallel queues

The default stream is one line: launch A, launch B, copy, launch C — each waits for the
last, and the copy engines and compute cores take turns sitting idle. The first move is
to notice that **independent jobs need no shared line**:

```cpp
// three independent images, three streams — the scheduler interleaves them on the SMs
fft<<<g,b,0,stream[i]>>>(img[i]);   filter<<<g,b,0,stream[i]>>>(img[i]);
```

The test for "can these share a stream or want separate ones?" is a single question:
**does B read what A wrote?** If no, give them different streams and let the hardware
overlap them.

## Producer → consumer: the one dependency that matters

When B *does* depend on A — but only at one point — you don't serialize everything. You
record an **event** after A and make B's stream wait on just that event. The CPU never
blocks; only stream `s2` does, and only until that exact moment in `s1`.

```cpp
decode<<<...,s1>>>();          // producer
cudaEventRecord(done, s1);
cudaStreamWaitEvent(s2, done); // consumer waits for *this point* in s1, nothing else
fft<<<...,s2>>>();
```

This is the atom every larger pattern is built from: *express the real dependency,
nothing more.*

## Pipeline — the production workhorse

A frame goes Disk → preprocess → **Upload → Compute → Download** → save. Done one frame
at a time, every stage waits for the previous. But the stages use *different hardware*
(PCIe in, SMs, PCIe out), so stagger them: while frame 1 computes, frame 2 is uploading
and frame 0 is downloading.

<svg class="dojo-diagram" viewBox="0 0 760 170" role="img" aria-label="Three frames pipelined across three engines; upload, compute, and download overlap in steady state.">
  <text class="mono" x="10" y="22">time →</text>
  <g class="mono">
    <text x="10" y="58">upload</text>
    <text x="10" y="98">compute</text>
    <text x="10" y="138">download</text>
  </g>
  <!-- frame 1 -->
  <rect class="fill-accent" x="120" y="42" width="120" height="24" rx="4"/><text class="mono" x="180" y="59" text-anchor="middle" fill="#0b1500">F1</text>
  <rect class="fill-accent" x="240" y="82" width="120" height="24" rx="4"/><text class="mono" x="300" y="99" text-anchor="middle" fill="#0b1500">F1</text>
  <rect class="fill-accent" x="360" y="122" width="120" height="24" rx="4"/><text class="mono" x="420" y="139" text-anchor="middle" fill="#0b1500">F1</text>
  <!-- frame 2 -->
  <rect class="fill-faint stroke-accent" x="240" y="42" width="120" height="24" rx="4"/><text class="mono" x="300" y="59" text-anchor="middle">F2</text>
  <rect class="fill-faint stroke-accent" x="360" y="82" width="120" height="24" rx="4"/><text class="mono" x="420" y="99" text-anchor="middle">F2</text>
  <rect class="fill-faint stroke-accent" x="480" y="122" width="120" height="24" rx="4"/><text class="mono" x="540" y="139" text-anchor="middle">F2</text>
  <!-- frame 3 -->
  <rect class="fill-faint stroke-faint" x="360" y="42" width="120" height="24" rx="4"/><text class="mono" x="420" y="59" text-anchor="middle">F3</text>
  <rect class="fill-faint stroke-faint" x="480" y="82" width="120" height="24" rx="4"/><text class="mono" x="540" y="99" text-anchor="middle">F3</text>
  <rect class="fill-faint stroke-faint" x="600" y="122" width="120" height="24" rx="4"/><text class="mono" x="660" y="139" text-anchor="middle">F3</text>
</svg>

In steady state all three engines are busy at once: PCIe-in, SMs, and PCIe-out overlap,
so the transfers are *hidden* behind compute. This is the Level 10 streams widget made
into a design — poke it again to feel the fill/drain cost shrink as frames increase:

<div data-dojo="streams"></div>

TensorRT, video codecs, microscopy, and sequencers all live on this pattern. The same
streams widget governs it.

## Double and triple buffering

The pipeline needs somewhere to put "the next frame" while the GPU chews on "this
frame." That's **double buffering**: two buffers, A and B — GPU computes on A while the
CPU (or the upload engine) fills B, then they swap. Every camera-acquisition pipeline
works this way.

```
GPU:  compute A │ compute B │ compute A   …
CPU:  fill   B  │ fill   A  │ fill   B     …   (swap each step)
```

If *upload is slower than compute*, two buffers aren't enough — the GPU finishes A and
the next buffer still isn't full. **Triple buffering** adds a third: one computing, one
uploading, one being filled, so no stage ever stalls waiting for a buffer. The cost is
memory; the payoff is never idling.

## Fan-out and fan-in

Two dependency shapes show up constantly. **Fan-out**: one result feeds several
independent consumers — run them in separate streams off the same producer event.
**Fan-in**: several independent producers must *all* finish before one consumer starts —
record an event per producer and have the consumer's stream wait on every one.

<svg class="dojo-diagram" viewBox="0 0 760 150" role="img" aria-label="Fan-out: one source branches to many. Fan-in: many sources merge before one sink.">
  <defs>
    <marker id="dojoarrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path class="fill-accent" d="M0,0 L6,3 L0,6 z"/></marker>
  </defs>
  <!-- fan-out -->
  <text class="mono" x="20" y="20">fan-out</text>
  <rect class="fill-accent" x="40" y="55" width="70" height="30" rx="4"/><text class="mono" x="75" y="74" text-anchor="middle" fill="#0b1500">FFT</text>
  <rect class="fill-faint stroke-accent" x="200" y="25" width="120" height="26" rx="4"/><text class="mono" x="260" y="42" text-anchor="middle">features</text>
  <rect class="fill-faint stroke-accent" x="200" y="60" width="120" height="26" rx="4"/><text class="mono" x="260" y="77" text-anchor="middle">thumbnail</text>
  <rect class="fill-faint stroke-accent" x="200" y="95" width="120" height="26" rx="4"/><text class="mono" x="260" y="112" text-anchor="middle">archive</text>
  <path class="stroke-faint" fill="none" marker-end="url(#dojoarrow)" d="M110,70 L195,38"/>
  <path class="stroke-faint" fill="none" marker-end="url(#dojoarrow)" d="M110,70 L195,73"/>
  <path class="stroke-faint" fill="none" marker-end="url(#dojoarrow)" d="M110,70 L195,108"/>
  <!-- fan-in -->
  <text class="mono" x="440" y="20">fan-in</text>
  <rect class="fill-faint stroke-accent" x="440" y="25" width="80" height="26" rx="4"/><text class="mono" x="480" y="42" text-anchor="middle">A</text>
  <rect class="fill-faint stroke-accent" x="440" y="60" width="80" height="26" rx="4"/><text class="mono" x="480" y="77" text-anchor="middle">B</text>
  <rect class="fill-faint stroke-accent" x="440" y="95" width="80" height="26" rx="4"/><text class="mono" x="480" y="112" text-anchor="middle">C</text>
  <rect class="fill-accent" x="640" y="55" width="70" height="30" rx="4"/><text class="mono" x="675" y="74" text-anchor="middle" fill="#0b1500">D</text>
  <path class="stroke-faint" fill="none" marker-end="url(#dojoarrow)" d="M520,38 L635,66"/>
  <path class="stroke-faint" fill="none" marker-end="url(#dojoarrow)" d="M520,73 L635,70"/>
  <path class="stroke-faint" fill="none" marker-end="url(#dojoarrow)" d="M520,108 L635,74"/>
</svg>

Fan-in is the easy one to get wrong: if D waits on only *one* of A/B/C, it can read a
buffer the other two haven't written yet — a classic missing-dependency race. The fix is
not a barrier; it's *recording all three events and waiting on all three.*

## Task graph — the generalization

Once a workload is a fixed DAG — `FFT → {filter, histogram} → classifier → output`,
re-run every frame — issuing it kernel-by-kernel pays the CPU launch cost
([Level 11](level11-multi-kernel.md)) *every* time. **CUDA Graphs** capture the whole
shape once and replay it with a single launch:

```cpp
cudaStreamBeginCapture(s, ...);   //   record the streams + events you just built
//   ... your normal launches & async copies, with their dependencies ...
cudaStreamEndCapture(s, &graph);
cudaGraphInstantiate(&exec, graph, ...);
cudaGraphLaunch(exec, s);         // replay the entire DAG, one dispatch
```

A graph is not a new pattern — it's *all the patterns above, frozen*. The producer/
consumer edges, the fan-out, the fan-in become the graph's dependency edges, and the
driver schedules them without the CPU in the loop. This is the natural successor to the
stream pipeline for *steady-state* work, and the backbone of [Level 14](level14-architecture.md).

## Persistent worker — when launches themselves are the cost

For huge numbers of *tiny, irregular* tasks, even graph replay has per-launch structure.
The **persistent kernel** flips it: launch *once*, with exactly enough blocks to fill
the GPU, and have them loop pulling work from a queue until it's empty.

```cpp
while (Task t = queue.pop())   // launched once; blocks never exit until drained
    process(t);
```

The per-iteration launch overhead disappears entirely. This needs blocks to coordinate
*across the whole grid* — the **cooperative-groups** grid barrier (`cg::this_grid();
grid.sync()`, with a cooperative launch) is what makes a single kernel able to do what
used to require relaunching. Inference servers, ray tracers, and graph-processing engines
run on this shape.

!!! warning "Pick the dependency, not the feature"
    Every pattern here is just "express the real dependency and nothing more." Reach for
    the heavier tool only when the profiler names the problem it solves: transfers not
    hidden → **pipeline + buffering**; CPU launch cost dominating a fixed DAG →
    **graphs**; per-launch overhead on tiny tasks → **persistent kernel**. Device-side
    work spawning (**dynamic parallelism**) exists for recursive/adaptive workloads but
    its overhead is real — it is rarely the answer; profile before committing.

## Your reps

- **Producer/consumer handoff** — take Level 10's chunked pipeline and split it into a
  *decode* stream and an *FFT* stream joined by one `cudaEventRecord` /
  `cudaStreamWaitEvent`. Confirm in `nsys` that the host never blocks between them.
- **Fan-in race hunt** — wire three independent kernels into a fourth, deliberately wait
  on only one event, and watch `compute-sanitizer --tool racecheck` (or wrong output)
  catch it. Then fix it by waiting on all three.
- **Capture a graph** — wrap a small fixed DAG in stream capture and replay it; measure
  the launch-overhead reduction in `nsys` against the un-captured version.
- For tensor-core and multi-GPU scaling (NCCL/NVSHMEM/MIG), see the
  [Architecture track](track-architecture.md).

??? question "Self-check"
    You fan three independent kernels A, B, C into a fourth kernel D, but D sometimes
    reads stale data. What's the bug, and why is `__syncthreads()` not the fix?
    *(D's stream waits on too few events — at most one of A/B/C — so it can start before
    the others finish writing. `__syncthreads()` only synchronizes threads within one
    block of one kernel; cross-kernel ordering is expressed with an event per producer
    and `cudaStreamWaitEvent` on all of them.)*

→ Continue to [Level 14 — Production Architecture](level14-architecture.md)
