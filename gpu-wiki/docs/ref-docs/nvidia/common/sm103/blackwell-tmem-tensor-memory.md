# Blackwell Tensor Memory (TMEM)

Reference on Blackwell's dedicated on-chip Tensor Memory (TMEM): what it is, its capacity and allocation model on sm_100/sm_103, how it differs from registers and SMEM, and why the FlashAttention-4 hd256 kernel overflows it.

**Provenance**: Public sources — [NVIDIA/CUTLASS](https://github.com/NVIDIA/cutlass) (`include/cute/arch/tmem_allocator_sm100.hpp`), [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) (`sm100_hd256_2cta_fmha_forward.py`), [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/). Deep-research pull: 2026-07-03.

## What it is

A new tier of storage introduced by Blackwell, dedicated to the tcgen05 tensor cores:

- **Location**: inside the SM, at the same level as SMEM (it is not a register, and not GMEM).
- **Purpose**: the **accumulator** for `tcgen05.mma` lives here, so it does not consume registers.
- **Access**: moved in/out of registers via `tcgen05.ld` / `tcgen05.st`.
- **Allocation**: explicitly allocated at runtime via `tcgen05.alloc` / `tcgen05.dealloc`.

## Hardware capacity (sm_100)

- **Per SM: 128 datapaths × 512 columns × 32-bit words = 256 KiB.**
- A row is a "datapath" (DP); a column is a "column"; the unit is 32-bit.
- Corresponding CUTLASS constants: `Sm100TmemCapacityColumns = 512`, `MAX_CAPACITY_BITS = Int<128*512*32>`.
- Location: `include/cute/arch/tmem_allocator_sm100.hpp`.

Kernels running on sm_103a currently treat the data exactly as sm_100 does, sharing the sm_100 arch layer (see [sm103-vs-sm100-differences.md](sm103-vs-sm100-differences.md)).

## Allocation model

- Unit: 32-column slice (`ColumnsPerAllocationSlice = 32`).
- `num_columns` must be a **power of two between 32 and 512** (32, 64, 128, 256, 512).
- PTX: `tcgen05.alloc.cta_group::[1|2].sync.aligned.shared::cta.b32`.
- Must be issued by **one fully-active warp** in the CTA.
- The 2-CTA case uses `Allocator2Sm`, the `cta_group::2` PTX variant.

## Differences from registers / SMEM

| | Register | SMEM | TMEM |
|---|---|---|---|
| Ownership | thread | CTA | SM (shareable across CTAs within a cluster) |
| Size | ~256/thread × 128 threads/warp × ... | 228 KB / SM (sm_100) | 256 KiB / SM |
| Access | direct instruction operand | `ld.shared` / `st.shared` | `tcgen05.ld` / `tcgen05.st` |
| Use | general | general shared, TMA staging | tensor-core accumulator only |
| Allocation | compile-time | dynamic launch attribute | runtime `tcgen05.alloc` |
| Release order | — | at CTA exit | must `tcgen05.dealloc` |

The benefit of TMEM: moving the accumulator out of registers frees a large number of registers for K/V load and online softmax → a single CTA can carry a larger tile / a deeper pipeline.

## Why FA4 hd256 overflows it

In the FA4 hd256 kernel (`sm100_hd256_2cta_fmha_forward.py`):

- The S accumulator occupies columns `[0, 256)` (hd256 × 32-bit).
- The O accumulator occupies columns `[256, 512)` (`tmem_o_offset=256`).
- **A single kernel fills the entire 512-column TMEM.**
- One CTA cannot hold it → it must run 2-CTA cooperatively (the code is always 2-CTA; there is no single-CTA variant).

**The real contention in the forward pass**: if another kernel on the same SM (NVFP4 MoE via CUTLASS, or GDN via fused_recurrent, both of which may also consume TMEM) allocates TMEM, the accumulator reads corrupted data → outputs of ~1e38 / NaN.

**Why it does not break offline**: offline runs only FA4, which owns the entire TMEM exclusively.

**Upstream fix**: when head_size > 128 and ≠ 192, vLLM's `fa_utils.py` forces a fallback to FA2 (see [vllm-sm103-attention-dispatch.md](vllm-sm103-attention-dispatch.md)). **Do not try to hand-fix the hd256 TMEM budget** — upstream has ruled it unsupported.

See also the pitfalls write-up: [sm100-vllm-fa4-hd256-seqused-pitfalls](../../../../pitfalls/nvidia/cutedsl/sm100-vllm-fa4-hd256-seqused-pitfalls.md).

## TMEM layout conventions (when writing your own kernel)

- Use `Allocator1Sm` or `Allocator2Sm` (in `tmem_allocator_sm100.hpp`).
- Partition the accumulator with column-offset constants like `tmem_s_offset` / `tmem_o_offset`; do not bake them into instruction immediates.
- Request the minimum necessary amount — do not allocate 512 "to leave yourself room": if another kernel on the same SM also allocates, you will overflow.
- Deallocation must be explicit (`tcgen05.dealloc`), otherwise you leak across kernels.

## References

- CUTLASS `tmem_allocator_sm100.hpp`: <https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/tmem_allocator_sm100.hpp>
- FA4 hd256 forward: <https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py>
- PTX ISA (tcgen05.alloc / dealloc / relinquish / TMEM access sections): <https://docs.nvidia.com/cuda/parallel-thread-execution/>
- Incident retrospective: local snapshot (path omitted).
