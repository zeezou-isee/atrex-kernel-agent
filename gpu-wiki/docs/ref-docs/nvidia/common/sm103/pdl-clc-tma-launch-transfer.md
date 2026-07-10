# PDL / CLC / TMA — Blackwell Launch & Transfer Features

Three launch/transfer features used on Blackwell: Programmatic Dependent Launch (PDL), Cluster Launch Control (CLC), and the Tensor Memory Accelerator (TMA), plus the CUDA launch attributes that drive them.

**Provenance**: Public sources — [NVIDIA/CUTLASS](https://github.com/NVIDIA/cutlass) (`copy_sm100_tma.hpp`), [djmmoss/ThunderKittens](https://github.com/djmmoss/ThunderKittens) fork (`nvfp4_b300_gemm.cu`). Deep-research pull: 2026-07-03.

## PDL — Programmatic Dependent Launch

- Introduced on Blackwell (actually present since Hopper; carried forward by Blackwell).
- Effect: before a grid has drained, the driver pre-warms the next dependent grid, hiding tail latency.
- Programming interface: the kernel sets `USE_PDL = true` (a macro in ThunderKittens), which underneath calls `cudaLaunchAttributeProgrammaticStreamSerialization`.
- Enabled by default in the `nvfp4_b300` GEMM (see [thunderkittens-nvfp4-k96-b300.md](thunderkittens-nvfp4-k96-b300.md)).

## CLC — Cluster Launch Control

See [cluster-2cta-cooperative-mma-clc.md](cluster-2cta-cooperative-mma-clc.md). Key points restated:

- Two cluster shapes, preferred + fallback; the driver uses the fallback when the remaining tiles cannot tile evenly.
- `clusterlaunchcontrol.try_cancel` is the PTX primitive for work-stealing.
- The kernel needs `cudaFuncAttributeNonPortableClusterSizeAllowed` to launch clusters of > 8 CTAs.

## TMA — Tensor Memory Accelerator

Introduced by Hopper, still used by Blackwell. CUTLASS `copy_sm100_tma.hpp` (~33 KB) is the sm_100-layer wrapper.

- Asynchronous bulk GMEM ↔ SMEM transfer, using neither registers nor the ordinary load units.
- Multi-dimensional tensor descriptors (via `cuTensorMapEncodeTiled` etc.).
- **multicast**: one GMEM read fans out to the SMEM of multiple CTAs in the cluster (present since Hopper).

## Relevant launch attributes at a glance

| Attribute | Function |
|---|---|
| `cudaFuncAttributeMaxDynamicSharedMemorySize` | Raise the dynamic SMEM ceiling |
| `cudaFuncAttributeNonPortableClusterSizeAllowed` | Allow clusters of > 8 CTAs |
| `cudaLaunchAttributeClusterDimension` | Static cluster shape |
| `cudaLaunchAttributePreferredClusterDimension` | CLC preferred (combined with fallback) |
| `cudaLaunchAttributeCooperative` | Allow cluster/grid sync |
| `cudaLaunchAttributeProgrammaticStreamSerialization` | Enable PDL |

## References

- CUTLASS TMA wrapper: <https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm100_tma.hpp>
- ThunderKittens `nvfp4_b300_gemm.cu` (usage of `LaunchConfig`).
