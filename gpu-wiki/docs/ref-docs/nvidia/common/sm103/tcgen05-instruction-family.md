# tcgen05.* Instruction Family

Blackwell's new tensor-core instruction set (PTX ISA 8.7+): every instruction is prefixed `tcgen05.` and decouples the tensor cores from Tensor Memory (TMEM). Covers the instruction inventory, the 7 `tcgen05.mma` kinds, per-call K, the 2-CTA cooperative variant, and a typical pipeline.

**Provenance**: Public sources — [NVIDIA/CUTLASS](https://github.com/NVIDIA/cutlass) (`mma_sm100_umma.hpp`, Blackwell functionality docs), [djmmoss/ThunderKittens](https://github.com/djmmoss/ThunderKittens) fork, [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-mma). Deep-research pull: 2026-07-03.

An entire new tensor-core instruction set introduced by Blackwell, living in PTX ISA 8.7+. Every instruction is prefixed `tcgen05.` and pulls the tensor cores apart from TMEM (see [blackwell-tmem-tensor-memory.md](blackwell-tmem-tensor-memory.md)).

## Instruction inventory

| Instruction | Function |
|---|---|
| `tcgen05.mma` | Execute U-MMA: `D = D + A @ B`; accumulator in TMEM, operands A/B in SMEM |
| `tcgen05.mma.sp` | Same, but A or B is 2:4 sparse |
| `tcgen05.ld` | Read from TMEM into registers |
| `tcgen05.st` | Write from registers into TMEM |
| `tcgen05.alloc` | Allocate TMEM (32-col slice; power-of-two 32–512 columns) |
| `tcgen05.dealloc` | Free TMEM (must be paired) |
| `tcgen05.wait::ld` / `tcgen05.wait::st` | Wait for TMEM ld/st completion (distinct from mma completion) |
| `tcgen05.commit` | Commit an mma group; synchronize together with `wait::mma` |
| `tcgen05.shift` | Shift within TMEM (used by some online-softmax patterns) |
| `tcgen05.relinquish_alloc_permit` | Return the alloc permit (TMEM exclusive ownership) |
| `tcgen05.fence` | Synchronization fence |

The full operand-type enumeration and modifiers for each instruction are in PTX ISA 9.7.17; below, only the key points confirmed in use on sm_103a are listed.

## The 7 `tcgen05.mma` kinds

| kind | Description | Throughput vs Hopper |
|---|---|---|
| `tf32` | FP32 accum, TF32 operand | 2× |
| `f16` | FP16/BF16 operand | 2× |
| `i8` | INT8 | 2× |
| `f8f6f4` | FP8/FP6/FP4 mixed (dense) | 2× |
| `mxf8f6f4.block_scale` | the above + MX block scale | 2× |
| `mxf4.block_scale` | MXFP4 (OCP) | **4×** |
| `mxf4nvf4.block_scale.scale_vec_size::[2X\|4X]` | NVFP4 or MXFP4, distinguished by `scale_vec_size` | **4×** |

Each has a dense and a `.sp` sparse form, and each has a `cta_group::1` and a `cta_group::2` form.

## K per tensor-core call

| dtype | K per call | vs Hopper |
|---|---|---|
| FP4 | **64 or 96** | new |
| FP8 | 32 | 16 → 32, ×2 |
| BF16 / FP16 | 16 | 8 → 16, ×2 |
| TF32 | 8 | 4 → 8, ×2 |

**K=96 is a Blackwell-exclusive advantage**: for FP4, K can be 64 or 96, making the K-dim tile 1.5× larger. See [nvfp4-k96-tcgen05-chunking.md](nvfp4-k96-tcgen05-chunking.md).

## 2-CTA cooperative (`cta_group::2`)

Two CTAs pool their tensor cores + TMEM and jointly run one large MMA tile (e.g. 256×256×128). Detailed constraints in [cluster-2cta-cooperative-mma-clc.md](cluster-2cta-cooperative-mma-clc.md).

## Typical pipeline

```
producer warpgroup:
    TMA load A/B to SMEM
    mbarrier arrive
consumer warpgroup (drives the tensor-core side):
    tcgen05.alloc D  (once)
    for k in K-tiles:
        wait producer barrier
        tcgen05.mma  D, A_smem_desc, B_smem_desc, D  (async, in-flight)
    tcgen05.commit
    tcgen05.wait::mma
    tcgen05.ld  D_reg, D
    (online-softmax / epilogue in registers)
    tcgen05.dealloc D
```

- `.alloc` is **exclusive**: another alloc in the same warp must be preceded by `.relinquish_alloc_permit`.
- `.mma` is **asynchronous**; issue several in a row to form an mma group, synchronized with `.commit` + `.wait::mma`.
- The ordering of `.ld` / `.st` relative to `.mma` is **not guaranteed**; explicit synchronization via fence / `.wait::ld` is required.

## Sparse `.mma.sp`

- 2:4 sparse pattern (same pattern as Ampere / Hopper).
- The SF vector size doubles in sparse mode (NVFP4 dense 16 → sparse 32; MXFP4 dense 32 → sparse 64).
- Throughput doubles again (dense f8 is 2× Hopper, so sparse f8 is 4×, and so on).

## References

- CUTLASS `mma_sm100_umma.hpp` (~82 KB, the main wrapper).
- ThunderKittens `include/ops/thread/mma/tcgen05.cuh` and `include/ops/group/mma/tcgen05.cuh`.
- CUTLASS Blackwell documentation: <https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md>
- PTX ISA: <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-mma>
