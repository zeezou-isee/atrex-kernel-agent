# ThunderKittens — NVFP4 K=96 on sm_103a (B300)

How ThunderKittens targets NVIDIA B300 / sm_103a (Blackwell Ultra, compute capability 10.3): the `SM103` build target sits alongside `SM100`/`SM120`, and a research fork adds sm_103a-specific kernels — most notably an NVFP4 GEMM with a **K=96 tcgen05 MMA primitive**, 2-CTA clusters, dynamic Cluster Launch Control (CLC), and PDL, plus two BF16 MLA-shape attention kernels hard-tuned to B300's 148 SMs.

## Provenance

- Upstream: <https://github.com/HazyResearch/ThunderKittens> (2.0, released 2026-01-11, introduced full Blackwell support including MXFP8 and NVFP4; `SM103` sits alongside `SM100`/`SM120` in the build system)
- Main sm_103a work fork: <https://github.com/djmmoss/ThunderKittens>
- Branch: `nvfp4-k96-tcgen05` (created 2026-04-23)
- HEAD: `ade0dbed85cc27c70687678d253df9f5497f2536` (2026-06-26, "chore: remove nvfp4 b300 handoff notes")
- Last substantive commit: 2026-06-25
- Local extract (path omitted): the bundle contains only the 3 SM103 kernel directories + `include/` + build glue.
- Pull / research date: 2026-07-03

## Operators Covered

| Category | Directory | Notes |
|---|---|---|
| GEMM | `kernels/gemm/nvfp4_b300/` | NVFP4, K=96 tcgen05 MMA, 2-CTA cluster, dynamic CLC, PDL |
| Attention (MHA fwd) | `kernels/attention/bf16_b300_mha_causal/` | BF16, **Dqk=192, Dvo=128** MLA shape, causal, `NUM_SM=148` |
| Attention (MHA fwd) | `kernels/attention/bf16_b300_mha_noncausal/` | Same shape, non-causal |
| DSL headers | `include/` | K=96-related additions to `ops/{group,thread}/mma/tcgen05.cuh` and `types/shared/descriptor.cuh` |
| Build glue | `kernels/common.mk` | `SM103` → `-gencode arch=compute_103a,code=sm_103a -DKITTENS_SM103` |

Only `nvfp4_b300` and `bf16_b300_mha_*` (three directories) are sm_103a-specific; `nvfp4_b200` and the other `_h100/_b200` directories belong to other arches.

## sm_103a-specific optimization points

### K=96 fp4 tcgen05 (the core innovation)

- Upstream ThunderKittens only exposes a K=64 fp4 MMA; this fork adds a **K=96 primitive**, making the K-dim tile of a single MMA 1.5x larger.
- Introducing commit: `218a033e` (2026-06-04, "feat: add NVFP4 K96 tcgen05 support"), touching 5 header files + 270 lines of tests, +473/-47.
- New API: `chunk_descriptor_k96(int mma_idx)` — combines two byte-offset descriptors at offsets `mma_idx*48` and `mma_idx*48+32`, packing the lower 14 bits of the next descriptor into bits `[16:29]` of the first descriptor, and setting bit 52 (absolute-address mode).
- Location: `include/types/shared/descriptor.cuh`, `include/ops/{group,thread}/mma/tcgen05.cuh`.

See [nvfp4-k96-tcgen05-chunking.md](nvfp4-k96-tcgen05-chunking.md).

### The hard constraint at the K=96 → 128B swizzle boundary

A K=96 chunk is 48 B and straddles the 128 B swizzle boundary. The fork enforces this with a set of static_asserts (`include/types/shared/descriptor.cuh` L92-99):

- Only packed NVFP4 (`fp4e2m1_2`) is supported.
- Only K-major (`!MN_major`) is supported.
- `ST::swizzle_bytes == 128` is required.

At the same time, `MMA_PER_TILE % 8 == 0` is also a hard static_assert in the kernel (see `kernels/gemm/nvfp4_b300/nvfp4_b300_gemm.cu` L17-42), for the same reason.

### 2-CTA cluster + dynamic Cluster Launch Control (CLC)

- `CLUSTER_SIZE = 2`
- `PREFERRED_CLUSTER_M = 4, PREFERRED_CLUSTER_N = 2`
- `FALLBACK_CLUSTER_M = 2, FALLBACK_CLUSTER_N = 1`
- CUTLASS 4.x-style dynamic cluster launch: when the preferred shape cannot tile the remaining work evenly, the driver falls back to the fallback shape.
- The launch uses `LaunchConfig` + `cudaFuncAttributeNonPortableClusterSizeAllowed`, passing both the preferred and fallback cluster dims.

**Why `PREFERRED_CLUSTER_N` defaults to 2 rather than 4:** CLC tiling requires `(N/Nb) % PREFERRED_CLUSTER_N == 0`. The author hit a crash with N=13824 (54 tiles) at `PREFERRED_CLUSTER_N=4` (54%4=2); switching to 2 fixed it (54%2=0). A comment in `kernels/gemm/nvfp4_b300/nvfp4_b300_gemm.cu` records this crash.

See [cluster-2cta-cooperative-mma-clc.md](cluster-2cta-cooperative-mma-clc.md).

### CLC-related follow-up commits

- `18e205c8` — add dynamic preferred/fallback cluster launch for `nvfp4_b300` (preferred 4x4 / fallback 2x1)
- `063eb5bb` — CLC work stealing, calling the `clusterlaunchcontrol.try_cancel` PTX directly
- `601379bc` — CLC configuration tuning

Together these three commits added +206 lines to `nvfp4_b300_gemm.cu`, covering region-based `blockIdx` assignment, runtime cluster-shape multicast, and barrier masks.

### PDL (Programmatic Dependent Launch)

`USE_PDL = true`: pre-warm the next grid wave, starting it up before the current grid has drained. See [pdl-clc-tma-launch-transfer.md](pdl-clc-tma-launch-transfer.md).

### `NUM_SM = 148` (hard-coded)

The attention kernels assume 148 SMs (the B300 count) and lay out the persistent grid directly for 148. **You cannot move this to B200 (132 SM) directly**: the persistent grid would be under-filled.

### MLA shape: `Dqk=192, Dvo=128`

Q and K have an extra RoPE-extended dim (so Dqk is 64 larger than Dvo), while V does not. Common in DeepSeek-V2/V3 and domestic large models. Other head dims will hit the `static_assert` in `config`.

## scale-address / SFID mode cheat sheet (`include/ops/thread/mma/tcgen05.cuh` L664-689)

| dtype / block | scale-addr stride | SFID |
|---|---|---|
| FP4E2M1 K=64, block_size=32 (MXFP4-like) | shared per MMA pair, `(i>>1)*M_offset` | alternates `idescs[0]` / `idescs[2]` |
| **FP4E2M1 K=96** | advances one aligned scale panel per MMA, `i*M_offset` | uses only `idescs[0]` |
| FP4E2M1 block_size=16 (NVFP4 with FP8E4M3 scales) | full `M_offset`/`N_offset` per MMA | always `SFID=0` (**not paired**) |
| MXFP8 K=64 | shared per 4 MMA, `(i>>2)*M_offset` | cycles `idescs[0..3]` |

## Build quick reference

```bash
# Standalone benchmark
cd kernels/gemm/nvfp4_b300 && make               # -> nvfp4_b300_gemm.out
./nvfp4_b300_gemm.out

# PyTorch extension (switch to the pytorch section in the Makefile)
make CONFIG=pytorch
python test_gemm.py
python test_quantize.py

# The two attention kernels (PyTorch extension by default)
cd kernels/attention/bf16_b300_mha_causal && make && python test.py
cd ../bf16_b300_mha_noncausal && make && python test.py
```

Hardware requirement: B300 / L20D / RTX Pro 6000 Blackwell (cc 10.3).

## Known gotchas / handoff notes

- The author's HANDOFF notes (deleted by `ade0dbe`) recorded: CLC leaves tensor duty at 55.6%, the persistent version at 55.4% — essentially a tie; the miss latency exposed by 4x4 multicast cannot be absorbed by the 4-stage pair-coupled pipeline — suggesting the pipeline depth may need to increase or switch to non-coupled.
- `NUM_SM = 148` is B300-specific; moving to B200 under-fills, and moving to a weaker workstation SKU (if the SM count differs) will too.
- The hard constraint again: K=96 NVFP4 runs only K-major, packed, `swizzle_bytes==128`; `MMA_PER_TILE % 8 == 0`.

## References

- Upstream README: <https://raw.githubusercontent.com/HazyResearch/ThunderKittens/main/README.md>
- SM103 target definition: `tests/Makefile` L35-44 (present in both upstream and this fork)
- TCGEN05 gate: `include/kittens.cuh` `#if defined(KITTENS_SM100) || defined(KITTENS_SM103)`
- K96 initial commit diff: <https://github.com/djmmoss/ThunderKittens/commit/218a033e16296568bcb03fd7878726b57b0b65e0>
- Sibling docs: [nvfp4-k96-tcgen05-chunking.md](nvfp4-k96-tcgen05-chunking.md), [cluster-2cta-cooperative-mma-clc.md](cluster-2cta-cooperative-mma-clc.md), [pdl-clc-tma-launch-transfer.md](pdl-clc-tma-launch-transfer.md), [tcgen05-instruction-family.md](tcgen05-instruction-family.md)
