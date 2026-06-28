# Architecture Deep Dive

> **The question:** *Your kernel has plenty of threads, but Nsight says the SM
> cannot issue instructions. What is the scheduler waiting for?*

The programming model says "threads." The hardware runs **warps**. An SM is a
small factory: warp schedulers choose ready warps, execution pipes do the work,
and registers/shared memory decide how many warps can live there at once.

Before reading on, make a prediction: if each warp waits hundreds of cycles on
memory, do you want more resident warps, more independent instructions per warp,
or fewer bytes moved?

<div data-dojo="sm-scheduler"></div>

## The SM as a factory

```
SM
├── warp schedulers      choose eligible warps
├── CUDA cores           scalar/vector arithmetic
├── Tensor Cores         matrix multiply-accumulate
├── load/store units     memory instructions
├── registers            per-thread working set
└── shared memory / L1   per-block scratchpad and cache
```

The scheduler does not care how many threads you launched. It asks: **which
resident warp has its next instruction ready?** If no warp is eligible, the issue
slot goes unused.

## Occupancy is a means, not the goal

Occupancy means resident warps divided by the hardware maximum. More occupancy
helps only when the extra warps hide latency. It can stop helping when:

- the kernel is already saturating memory bandwidth;
- each warp has enough independent work to cover its own latency;
- higher occupancy requires smaller tiles or more spills.

The useful question is: **what limits eligible warps?**

| Limiter | What to inspect | CUDA idea |
|---------|-----------------|-----------|
| Register pressure | registers/thread, spills, achieved occupancy | fewer live values can fit more warps |
| Shared memory/block | shared allocation, blocks/SM | larger tiles trade occupancy for reuse |
| Memory latency | warp stall reasons, global load efficiency | coalesce, cache, tile, or add independent work |
| Barriers | stall barrier, `__syncthreads()` placement | blocks wait for the slowest cooperating thread |

## Transactions, caches, and banks

A warp load is not 32 independent scalar loads. The memory system groups lane
requests into transactions. Contiguous lanes touching contiguous addresses waste
little bandwidth; strided lanes pay for bytes they do not use.

Caches help when data is reused. Shared memory helps when you can stage a tile and
control reuse yourself. Shared memory also has a failure mode: **bank conflicts**.
Thirty-two lanes hitting 32 banks is fast. Many lanes hitting one bank serialize.

## Tensor Cores and low-level inspection

Tensor Cores are just another execution pipe, but a specialized one: matrix
multiply-accumulate. You usually reach them through cuBLAS, cuBLASLt, CUTLASS, or
cuDNN before writing WMMA directly.

PTX and SASS are tools for explaining evidence. Use them after Nsight points to
instruction mix, register pressure, spills, or unexpected memory operations. PTX
shows compiler intent; SASS is what actually runs.

## Scaling out: multiple GPUs

One SM is a factory; one GPU is a building of them; a node is several buildings wired
together. The bandwidth between them is the whole story — NVLink (GPU↔GPU, ~100s GB/s)
is far faster than PCIe (host↔GPU, ~tens GB/s), so the design question is always *how
little can cross the slow link.* The tools that manage that crossing:

| Tool | What it is | When it's the answer |
|------|------------|----------------------|
| **NCCL** | Tuned collectives — all-reduce, broadcast, all-gather — that pick the right ring/tree over the actual NVLink/PCIe topology | Scaling DL training across GPUs/nodes. Use it; don't hand-roll cross-GPU reductions |
| **NVSHMEM** | A PGAS model: a GPU reads/writes another GPU's memory directly, one-sided | Fine-grained, irregular cross-GPU access where collectives are too coarse |
| **Peer-to-peer (P2P)** | One GPU dereferences another's memory when topology + permissions allow | Two GPUs on the same NVLink/PCIe root sharing buffers without a host bounce |
| **MIG** | Partition one big GPU into isolated instances with their own SMs/memory | Multi-tenant *serving* — predictable isolation, not raw throughput |

The mental model is the same one occupancy taught, one level up: a collective is "fast"
only when it keeps the *interconnect* busy with useful bytes, the way a kernel keeps the
SMs busy with eligible warps. Profile cross-GPU traffic in `nsys` before assuming a
collective is the bottleneck.

## Your reps

- Predict which slider setting in the widget will waste issue slots, then move it.
- Compile one kernel with low and high register pressure. Predict occupancy before
  checking Nsight Compute.
- For a stride experiment, predict memory transactions before looking at global
  load efficiency.
- If you have ≥2 GPUs: run an **NCCL all-reduce** of a vector and reason about whether
  it travels over NVLink or PCIe — predict the bandwidth before profiling it.

??? question "Self-check"
    A kernel has 35% occupancy and 95% of peak DRAM bandwidth. Should you chase
    higher occupancy? *(Not first. The SM has enough warps to saturate memory.
    Move fewer bytes or reuse them more.)*
