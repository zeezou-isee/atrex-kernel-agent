# NVIDIA CUTLASS — Blackwell Arch Layer for sm_103a (B300)

How CUTLASS targets NVIDIA B300 / sm_103a (Blackwell Ultra, compute capability 10.3): there is no dedicated `sm_103` arch layer — sm_103a reuses the sm_100 arch primitives and kernel builders, and any sm_103a-specific tuning has to be done in your own kernel.

## Provenance

- Repository: <https://github.com/NVIDIA/cutlass>
- Blackwell functionality docs: <https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md>
- Key directories: `include/cute/arch/` (arch primitives), `include/cutlass/gemm/collective/builders/sm100_common.inl` (kernel construction helpers)
- Related local reference: the internal pai-vllm fork (URL omitted) embeds a copy of `csrc/cutlass/` containing `cluster_sm100` / `tmem_allocator_sm100` (local snapshot, path omitted).
- Pull / research date: 2026-07-03

## Blackwell Operators Covered

CUTLASS is not an "operator implementation" but rather a set of **arch primitives + kernel builders**. The sm_100 layer files:

| File | Approx. size | Coverage |
|---|---:|---|
| `cluster_sm100.hpp` | — | cluster launch API, preferred + fallback, cta_group |
| `copy_sm100.hpp` | 365 KB | tcgen05.ld/st, generic copy |
| `copy_sm100_tma.hpp` | 33 KB | Blackwell TMA descriptor |
| `mma_sm100.hpp` | — | tcgen05.mma top level |
| `mma_sm100_desc.hpp` | — | tcgen05.mma descriptor |
| `mma_sm100_umma.hpp` | 82 KB | U-MMA / tcgen05.mma instruction wrappers (the bulk) |
| `simd_sm100.hpp` | — | SIMD mnemonics |
| `tmem_allocator_sm100.hpp` | — | TMEM allocator |

**There is no standalone `sm_103` arch file** — sm_103a reuses the sm_100 arch layer. To tune specifically for sm_103a you are on your own: inside your kernel, use `#if __CUDA_ARCH__ == 1030` to pick different tile / stage / cluster shapes, but the underlying PTX is the same sm_100 set. See [sm103-vs-sm100-differences.md](sm103-vs-sm100-differences.md).

## tcgen05.mma kinds supported on sm_100 (FP8 family)

7 kinds, each with a dense and a `.sp` (sparse) variant, and each with `cta_group::[1|2]`:

| kind | Tensor-core throughput vs. comparable Hopper |
|---|---|
| `tf32` | 2x |
| `f16` | 2x |
| `i8` | 2x |
| `f8f6f4` | 2x |
| `mxf8f6f4.block_scale` | 2x |
| `mxf4.block_scale` | 4x |
| `mxf4nvf4.block_scale.scale_vec_size::[2X\|4X]` | 4x |

The last two block-scale kinds double throughput again — these are the channels you must use for NVFP4/MXFP4 GEMM on Blackwell.

**K per tensor-core call:**

| dtype | K per call |
|---|---|
| FP4 | **64 or 96** |
| FP8 | 32 |
| BF16 / FP16 | 16 |
| TF32 | 8 |

Compared to Hopper: FP8/BF16/TF32 are 32/16/8 respectively, so Blackwell doubles each. See [tcgen05-instruction-family.md](tcgen05-instruction-family.md).

## NVFP4 vs MXFP4 distinction

| Type | scale dtype | SF vec (dense/sparse) | OCP-compliant |
|---|---|---|---|
| `nv_float4_t` (NVFP4) | `float_ue4m3_t` (FP8 E4M3) | **16 / 32** | No |
| `mx_float4_t` (MXFP4) | `float_ue8m0_t` (FP8 E8M0) | 32 / 64 | Yes |

The scale factor is applied per block along the K dimension; the PTX `scale_vec_size::[2X|4X]` modifier encodes the SF-vec difference. **The MMA K-tile is still 64 for both**; the SF vec size determines how many SF entries fill that tile. See [nvfp4-k96-tcgen05-chunking.md](nvfp4-k96-tcgen05-chunking.md).

## MmaTileShape supported for NVFP4 GEMM

**TN layout only, K fixed at 256, A/B alignment = 32 elements:**

| Variant | dispatch policy | MmaTileShape |
|---|---|---|
| 1SM | `KernelTmaWarpSpecialized1SmNvf4Sm100` | `128 × {128, 192, 256} × 256` |
| 2SM | `KernelTmaWarpSpecialized2SmNvf4Sm100` | `256 × {128, 192, 256} × 256` |

## Cluster constraints for 2-CTA cooperative MMA

`cta_group::2` / `KernelTmaWarpSpecialized2SmSm100` requires:

- The M mode of `ClusterShape_MNK` must satisfy `% 2 == 0`.
- Legal shapes: `cute::Shape<_2|_4, [_1|_2|_4], _1>`.

`include/cutlass/gemm/collective/builders/sm100_common.inl` L205 / L241:

```cpp
static_assert(cute::size<0>(cluster_shape_mnk) % 2 == 0,
              "Cluster shape not divisible by MMA size");
```

This is the mechanism-layer origin of "two CTAs pool their tensor cores + TMEM to run a single large tile (e.g. 256×256×128)". See [cluster-2cta-cooperative-mma-clc.md](cluster-2cta-cooperative-mma-clc.md).

## cluster_sm100.hpp — preferred + fallback launch

```cpp
initialize_preferred_cluster_launch(
    kernel_function,
    grid_dims,
    cluster_dims_preferred,
    cluster_dims_fallback);
```

Runtime validation (header L49-100):

- Total cluster ≤ 32 CTA (hardware constraint)
- `grid_dims % cluster_dims_preferred == 0`
- `cluster_dims_preferred % cluster_dims_fallback == 0`
- Both clusters share the same z depth

See [pdl-clc-tma-launch-transfer.md](pdl-clc-tma-launch-transfer.md).

## TMEM Allocator

`tmem_allocator_sm100.hpp`:

- Hardware-capacity comment: `// 128 DP x 512 COL x uint32_t-addressing`
- `MAX_CAPACITY_BITS = Int<128*512*32>`
- `ColumnsPerAllocationSlice = 32`
- `Sm100TmemCapacityColumns = 512`
- PTX: `tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32`
- **`num_columns` must be a power of two between 32 and 512.**
- The allocation must be issued by **one fully-active warp** in the CTA; `Allocator2Sm` handles the `cta_group::2` cooperative version.

See [blackwell-tmem-tensor-memory.md](blackwell-tmem-tensor-memory.md).

## Relationship to writing your own sm_103a kernel

- Need tcgen05.mma → use the wrappers in `mma_sm100_umma.hpp`.
- Need TMA → `copy_sm100_tma.hpp`.
- Need 2-CTA cluster + fallback → the API in `cluster_sm100.hpp`.
- Need the tile-shape constraints for NVFP4 GEMM → copy the builder in `sm100_common.inl`, or look at CUTLASS's own Blackwell examples (examples/77+, not extracted here).
- **You cannot find sm_103a-specific tuning directly**; if you want an sm_103a-only fast path, add a `constexpr` branch at the kernel level yourself.

## References

- CUTLASS repository: <https://github.com/NVIDIA/cutlass>
- Blackwell functionality docs: <https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md>
- Sibling hardware notes: [sm103-vs-sm100-differences.md](sm103-vs-sm100-differences.md), [blackwell-tmem-tensor-memory.md](blackwell-tmem-tensor-memory.md), [tcgen05-instruction-family.md](tcgen05-instruction-family.md), [cluster-2cta-cooperative-mma-clc.md](cluster-2cta-cooperative-mma-clc.md), [nvfp4-k96-tcgen05-chunking.md](nvfp4-k96-tcgen05-chunking.md), [pdl-clc-tma-launch-transfer.md](pdl-clc-tma-launch-transfer.md)
