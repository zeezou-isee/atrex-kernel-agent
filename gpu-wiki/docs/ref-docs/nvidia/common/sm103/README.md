# Blackwell Ultra (SM103 / B300) — Kernel Research Reference

Cross-source reference for optimizing kernels on NVIDIA **B300 / sm_103a** (Blackwell Ultra, compute capability 10.3; also L20D / RTX Pro 6000 Blackwell). Distilled from a 2026-07-03 deep-research pass over public repositories (NVIDIA/cutlass, Dao-AILab & vllm-project flash-attention, HazyResearch & djmmoss ThunderKittens, vLLM) plus the PTX ISA and CUTLASS Blackwell docs.

**Key takeaway:** sm_103a has **no dedicated arch layer** in any of these stacks — CUTLASS, FA4, and vLLM all dispatch B300 through the **sm_100** code path. Any sm_103a-specific tuning must be done in your own kernel (e.g. `#if __CUDA_ARCH__ == 1030`). The djmmoss ThunderKittens fork is the most concentrated true sm_103a reference (K=96 NVFP4 tcgen05 GEMM + BF16 MLA attention).

---

## Hardware & ISA

| File | Description |
|------|------|
| [sm103-vs-sm100-differences.md](sm103-vs-sm100-differences.md) | Confirmed differences between the two Blackwell targets (sm_100a/B200 vs sm_103a/B300 Ultra) and three rules for portable kernel authoring |
| [tcgen05-instruction-family.md](tcgen05-instruction-family.md) | The `tcgen05.*` instruction set: inventory, the 7 MMA kinds, per-call K (incl. K=96), 2-CTA cooperative mode, canonical producer/consumer pipeline |
| [blackwell-tmem-tensor-memory.md](blackwell-tmem-tensor-memory.md) | Tensor Memory: 128 DP × 512 col capacity, allocation model, register/SMEM comparison, and why FA4 hd256 overflows TMEM under concurrent kernels |
| [cluster-2cta-cooperative-mma-clc.md](cluster-2cta-cooperative-mma-clc.md) | Clusters, `cta_group::2` cooperative MMA, and Cluster Launch Control (preferred+fallback shapes, `try_cancel` work-stealing) with common traps |
| [nvfp4-k96-tcgen05-chunking.md](nvfp4-k96-tcgen05-chunking.md) | NVFP4 vs MXFP4 block-scaling, the shared `mxf4nvf4` MMA kind, and the djmmoss K=96 chunking innovation (48B/128B-swizzle constraints, absolute-address descriptor, SFID patterns) |
| [pdl-clc-tma-launch-transfer.md](pdl-clc-tma-launch-transfer.md) | Blackwell launch/transfer features (PDL, CLC, TMA) and the CUDA launch attributes that enable them |

## Operator & library notes (per repository)

| File | Description |
|------|------|
| [thunderkittens-nvfp4-k96-b300.md](thunderkittens-nvfp4-k96-b300.md) | ThunderKittens `SM103` target + djmmoss `nvfp4-k96-tcgen05` fork: K=96 NVFP4 tcgen05, 2-CTA + dynamic CLC/PDL, 148-SM-tuned BF16 MLA attention — the most concentrated sm_103a reference implementation |
| [flash-attention-4-sm103-dispatch.md](flash-attention-4-sm103-dispatch.md) | FA4 has no sm103 build target; all 10.x devices dispatch into the SM100 bucket, so B300 runs sm_100 artifacts including the hd256 2-CTA kernel and its 512-column TMEM-contention garbage bug |
| [cutlass-blackwell-sm103-arch-layer.md](cutlass-blackwell-sm103-arch-layer.md) | CUTLASS has no dedicated sm_103 arch layer; sm_103a reuses the sm_100 tcgen05/TMA/TMEM primitives, cluster launch API, and NVFP4 GEMM tile-shape / 2-CTA constraints |
| [vllm-sm103-attention-dispatch.md](vllm-sm103-attention-dispatch.md) | vLLM ships no sm_103a kernel; its `fa_utils.py` head_size guard (>128, ≠192 → FA4 downgrade to FA2) is the critical gate that prevents the hd256 garbage, plus NVFP4 MoE grouped-GEMM kernels seen in traces |

## Related

- FA4 hd256 TMEM-contention garbage pitfall: [../../../../pitfalls/nvidia/cutedsl/sm100-vllm-fa4-hd256-seqused-pitfalls.md](../../../../pitfalls/nvidia/cutedsl/sm100-vllm-fa4-hd256-seqused-pitfalls.md)
- CuTeDSL FA4 hd256 prefill optimization journey on B300: [../../cutedsl/sm103/sm103-fa4-hd256-prefill-optimization.md](../../cutedsl/sm103/sm103-fa4-hd256-prefill-optimization.md)
- Blackwell datacenter (SM100) reference collection: [../sm100/](../sm100/)
