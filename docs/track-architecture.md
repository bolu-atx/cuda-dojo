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

## Your reps

- Predict which slider setting in the widget will waste issue slots, then move it.
- Compile one kernel with low and high register pressure. Predict occupancy before
  checking Nsight Compute.
- For a stride experiment, predict memory transactions before looking at global
  load efficiency.

??? question "Self-check"
    A kernel has 35% occupancy and 95% of peak DRAM bandwidth. Should you chase
    higher occupancy? *(Not first. The SM has enough warps to saturate memory.
    Move fewer bytes or reuse them more.)*
