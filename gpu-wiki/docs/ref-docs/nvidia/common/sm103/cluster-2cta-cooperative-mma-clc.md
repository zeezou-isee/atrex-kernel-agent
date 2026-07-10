# Cluster / 2-CTA Cooperative MMA / CLC

Blackwell clusters, the `cta_group::2` cooperative MMA, and Cluster Launch Control (CLC): constraints, when to use 2-CTA (and when not), the preferred+fallback shape mechanism, `clusterlaunchcontrol.try_cancel` work-stealing, and common traps.

**Provenance**: Public sources — [NVIDIA/CUTLASS](https://github.com/NVIDIA/cutlass) (`cluster_sm100.hpp`, `sm100_common.inl`), [djmmoss/ThunderKittens](https://github.com/djmmoss/ThunderKittens) fork `nvfp4_b300` kernel (work-stealing commit `063eb5bb`), [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/). Deep-research pull: 2026-07-03.

## Cluster basics (present since Hopper; carried forward and extended by Blackwell)

- A cluster is a group of co-resident CTAs (Cooperative Thread Arrays, i.e. CUDA blocks).
- CTAs within a cluster can read/write each other's SMEM through **DSMEM** (distributed shared memory).
- Clusters are SM-affine (the scheduler guarantees that CTAs of the same cluster land on physically adjacent SMs, for good DSMEM bandwidth).

## Cluster constraints on Blackwell

- **Total cluster ≤ 32 CTAs** (hardware upper bound).
- CUTLASS `initialize_preferred_cluster_launch` runtime validation:
  - `grid_dims % cluster_dims_preferred == 0`
  - `cluster_dims_preferred % cluster_dims_fallback == 0`
  - preferred / fallback share the z depth
- See [cutlass-blackwell-sm103-arch-layer.md](cutlass-blackwell-sm103-arch-layer.md), item F12.

## 2-CTA cooperative MMA (`cta_group::2`)

Blackwell's `tcgen05.mma` has two `cta_group` modes:

- `cta_group::1` — one CTA runs independently.
- `cta_group::2` — **two CTAs pool their tensor cores + TMEM and jointly run one large MMA tile.**

Usage:

- The kernel policy uses `KernelTmaWarpSpecialized2SmSm100`.
- `cluster_shape.M % 2 == 0` (hard constraint, static_assert).
- Legal `ClusterShape_MNK`: `cute::Shape<_2|_4, [_1|_2|_4], _1>`.
- The MMA tile is typically `256 × {128, 192, 256} × K` (K=256 for NVFP4).

**Why 2-CTA**:

1. A larger MMA tile amortizes tensor-core launch overhead.
2. TMEM pooling — a single CTA's 512 columns are not enough (e.g. FA4 hd256 must be 2-CTA; see [blackwell-tmem-tensor-memory.md](blackwell-tmem-tensor-memory.md)).
3. DSMEM lets the two CTAs split the A/B load, reducing GMEM requests.

**Why not always 2-CTA**:

- Many hard constraints (cluster_M % 2 == 0, strict shape limits).
- When the remaining tiles do not tile evenly, one CTA's slot is wasted (hence the CLC fallback).

## Cluster Launch Control (CLC)

Introduced by Blackwell; allows the cluster shape to be decided **at runtime**.

### preferred + fallback shape

- The kernel declares two cluster shapes: preferred + fallback.
- The driver tries the preferred tile for the remaining work; when the remainder cannot tile evenly, it uses the fallback.
- Typical example (ThunderKittens `nvfp4_b300` GEMM): preferred=4×2, fallback=2×1.

### `clusterlaunchcontrol.try_cancel`

A single PTX instruction that lets a CTA voluntarily abandon the tile it was assigned, returning it to the pool for another CTA to steal — **work-stealing**.

- Added in the ThunderKittens fork commit `063eb5bb`.
- Effect: reduced tail latency; in a persistent kernel, idle CTAs help drain the queue.
- The author's HANDOFF notes record: CLC work-stealing lifted tensor duty from 55.4% (persistent) to 55.6% (CLC) — a nearly negligible gain; the main bottleneck is the 4×4 multicast miss latency, which the 4-stage pair-coupled pipeline cannot absorb.

### Hooking into the CUDA API

Launch with:

```cpp
cudaLaunchAttribute attrs[2];
attrs[0].id = cudaLaunchAttributeClusterDimension;         // preferred
attrs[1].id = cudaLaunchAttributePreferredClusterDimension;// (naming varies)
cudaLaunchConfig_t cfg = {...};
cfg.attrs = attrs;
cfg.numAttrs = 2;
```

The kernel also needs `cudaFuncAttributeNonPortableClusterSizeAllowed` (otherwise a cluster of > 8 CTAs is rejected).

## Common traps

1. `PREFERRED_CLUSTER_N` must evenly divide `N / Nb` (the tile count). A case that was hit: N=13824, Nb=? → 54 tiles; `PREFERRED_CLUSTER_N=4` crashes (54%4=2), `=2` works (54%2=0). See [thunderkittens-nvfp4-k96-b300.md](thunderkittens-nvfp4-k96-b300.md).
2. Relationship between `MMA_PER_TILE` and K: K=96 NVFP4 must satisfy `% 8 == 0` (swizzle boundary).
3. The 4×4 multicast deep end: when the pipeline is not deep enough, misses are directly exposed — with stages ≤ 4 there is basically no hope of saturating bandwidth.

## References

- CUTLASS `cluster_sm100.hpp`.
- CUTLASS `sm100_common.inl` (the cluster_M % 2 static_assert).
- ThunderKittens `nvfp4_b300` kernel (CLC + fallback).
- PTX ISA `clusterlaunchcontrol` section: <https://docs.nvidia.com/cuda/parallel-thread-execution/>
