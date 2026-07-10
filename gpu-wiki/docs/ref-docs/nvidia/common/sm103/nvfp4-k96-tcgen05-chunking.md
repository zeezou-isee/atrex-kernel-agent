# NVFP4 and the K=96 tcgen05 Chunk

NVFP4 (NVIDIA's non-OCP E2M1 4-bit float with block scaling) versus MXFP4, how both map onto the single `mxf4nvf4.block_scale` MMA kind via `scale_vec_size`, and the K=96 chunking innovation in the djmmoss ThunderKittens fork: the 48B-chunk vs 128B-swizzle hard constraints, the absolute-address descriptor mode, and the SFID scale-address patterns.

**Provenance**: Public sources — [NVIDIA/CUTLASS](https://github.com/NVIDIA/cutlass) Blackwell functionality docs, [djmmoss/ThunderKittens](https://github.com/djmmoss/ThunderKittens) fork (`descriptor.cuh`, `tcgen05.cuh`; K96 initial commit [`218a033`](https://github.com/djmmoss/ThunderKittens/commit/218a033e16296568bcb03fd7878726b57b0b65e0)), [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-mma-block-scaling). Deep-research pull: 2026-07-03.

## What NVFP4 is

- E2M1 packed 4-bit float (NVIDIA variant; **does not** follow the OCP standard).
- **Block scale**: every 16 elements share one scale; the scale uses FP8 E4M3 (`float_ue4m3_t`).
- CUTLASS type: `nv_float4_t`.

Compared with MXFP4 (OCP standard):

- every 32 elements share one scale
- the scale uses FP8 E8M0 (`float_ue8m0_t`)
- CUTLASS type: `mx_float4_t`

Comparison table: see [cutlass-blackwell-sm103-arch-layer.md](cutlass-blackwell-sm103-arch-layer.md), item F9.

## Encoding at the PTX level

The `tcgen05.mma` kind:

- `mxf4nvf4.block_scale.scale_vec_size::[2X|4X]`
- **A single instruction** supports both NVFP4 and MXFP4; the `scale_vec_size` modifier decides:
  - `2X` → each K-tile has an SF covering 32 elements (dense) / 64 (sparse) — MXFP4
  - `4X` → each K-tile has an SF covering 16 elements (dense) / 32 (sparse) — **NVFP4**

**The MMA K-tile is 64 for both** (or 96, see below); the only difference is the SF vector size.

## K per NVFP4 MMA call

- **K = 64** — supported all along by upstream ThunderKittens.
- **K = 96** — the core innovation of the djmmoss fork: the K-dim tile is 1.5× larger, so theoretical compute is also 1.5× larger.

## The hard constraints of K=96 (48B chunk × 128B swizzle boundary)

- Each K=96 NVFP4 chunk = 48 B (96 × 4-bit).
- A 128 B swizzle atom cannot hold an integer number of 48 B chunks (48 → 96 → 144 is needed to reach the 128B boundary).
- Solution: **absolute address mode**, encoded by bit 52 (`lbo_mode`) of the shared-memory matrix descriptor.
- Reference: PTX ISA 9.7.17.3.1.2.

### Implementation in the fork

`include/types/shared/descriptor.cuh` L92–99, the hard constraints:

```cpp
static_assert(is_fp4e2m1_2, "K=96 requires packed NVFP4 (fp4e2m1_2)");
static_assert(!is_MN_major, "K=96 requires K-major");
static_assert(ST::swizzle_bytes == 128, "K=96 requires 128B swizzle atom");
```

Runtime (L103–111):

```cpp
lbo_byte = base + (gap < 48 ? gap : 32);
desc |= 1ull << 52;                    // absolute address mode
desc |= ((next_desc & 0x3FFF) << 16);  // pack the low 14 bits of the next descriptor
```

**K=96 is allowed only when all three of packed-NVFP4, K-major, and 128B swizzle hold simultaneously**; if any one is unmet, it is a compile error.

## MMA_PER_TILE hard constraint

`kernels/gemm/nvfp4_b300/nvfp4_b300_gemm.cu` L17–42:

```cpp
static_assert(MMA_PER_TILE % 8 == 0, ...);
```

Same reason as the 128B swizzle atom — one 128B atom packs 8 MMAs' worth of A or B.

## scale-address / SFID patterns (fork, by precision)

`include/ops/thread/mma/tcgen05.cuh` L664–689:

| dtype / block | scale-addr stride | SFID |
|---|---|---|
| FP4E2M1 K=64, block_size=32 (MXFP4-like) | shared per MMA pair, `(i>>1)*M_offset` | alternate `idescs[0]` / `idescs[2]` |
| **FP4E2M1 K=96** | advance one aligned scale panel per MMA, `i*M_offset` | `idescs[0]` only |
| FP4E2M1 block_size=16 (NVFP4 with FP8E4M3 scales) | one full `M_offset`/`N_offset` per MMA | constant `SFID=0` (**not paired**) |
| MXFP8 K=64 | shared per 4 MMAs, `(i>>2)*M_offset` | cycle `idescs[0..3]` |

**K=96 uses SFID=0 exclusively**, not the paired SFID switching of K=64 — a direct consequence of the fact that, at the hardware level, one K=96 MMA consumes exactly one scale panel.

## GEMM tile shape (NVFP4)

The legal CUTLASS combinations (K fixed at 256, TN, A/B alignment=32):

- 1SM (`KernelTmaWarpSpecialized1SmNvf4Sm100`): `128 × {128, 192, 256} × 256`
- 2SM (`KernelTmaWarpSpecialized2SmNvf4Sm100`): `256 × {128, 192, 256} × 256`

## MXFP8 and other new formats (not fully covered)

- MXFP8: E4M3 or E5M2 + E8M0 scale (per 32 elements); the ThunderKittens fork also runs MXFP8 with K=64, scale shared per 4 MMAs (SFID cycles 0..3).
- FP8 E4M3 / E5M2 dense: `tcgen05.mma` kind `f8f6f4`, K=32.
- FP6 / FP4 dense (no block scale): also uses the `f8f6f4` kind.

## References

- CUTLASS Blackwell documentation: <https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md>
- ThunderKittens fork `descriptor.cuh` (K96 hard constraints).
- ThunderKittens fork `tcgen05.cuh` (SFID patterns).
- K96 initial commit: <https://github.com/djmmoss/ThunderKittens/commit/218a033e16296568bcb03fd7878726b57b0b65e0>
- PTX ISA scale_vec_size: <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-mma-block-scaling>
