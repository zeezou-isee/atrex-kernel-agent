# sm_103a vs sm_100a — Differences Between the Two Blackwell Targets

How NVIDIA's two Blackwell compilation targets (sm_100a on B200, sm_103a on B300 / Blackwell Ultra, compute capability 10.3) differ in practice, and the three things that actually affect kernel authoring.

**Provenance**: Public sources — [NVIDIA/CUTLASS](https://github.com/NVIDIA/cutlass), [djmmoss/ThunderKittens](https://github.com/djmmoss/ThunderKittens) fork (work-stealing commit `063eb5bb`). Deep-research pull: 2026-07-03.

## Terminology

- **sm_100a** — Blackwell datacenter, B200.
- **sm_103a** — Blackwell workstation / Ultra: B300, L20D, RTX Pro 6000 Blackwell (compute capability 10.3).

## Confirmed (backed by a primary source)

### Compilation target

- CUTLASS has **no standalone `sm_103` arch file** — sm_103a reuses the entire sm_100 arch layer (see [cutlass-blackwell-sm103-arch-layer.md](cutlass-blackwell-sm103-arch-layer.md), item F13).
- The ThunderKittens build system carries a distinct `SM103` target: `-gencode arch=compute_103a,code=sm_103a -DKITTENS_SM103`, and combines it with `#if defined(KITTENS_SM100) || defined(KITTENS_SM103)` to route the TCGEN05 primitives through to sm_103a.

### tcgen05 instruction family

- The full set of `tcgen05.mma` kinds (7 kinds, dense + sparse, cta_group 1/2) is available on sm_103a — generated both via the sm_100 build product and behind the `KITTENS_SM103` gate. See [tcgen05-instruction-family.md](tcgen05-instruction-family.md).

### 2-CTA cooperative MMA

- The cluster-shape constraints (M mode % 2 == 0; legal shape `<_2|_4, [_1|_2|_4], _1>`) apply on sm_103a (the fork uses preferred=4x4 / fallback=2x1 directly). See [cluster-2cta-cooperative-mma-clc.md](cluster-2cta-cooperative-mma-clc.md).
- A single cluster of ≤ 32 CTAs is the hardware upper bound.

### CLC and dynamic cluster launch

- The preferred + fallback API works on sm_103a. The fork uses `cudaFuncAttributeNonPortableClusterSizeAllowed` plus a `LaunchConfig` carrying two sets of cluster dims.
- The `clusterlaunchcontrol.try_cancel` PTX is available on sm_103a (the fork's work-stealing commit `063eb5bb` calls it directly).

## Three things that actually affect kernel authoring

1. **Do not hard-code `sm_100a` in the kernel.** At minimum, write `#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030`.
2. **Do not hard-code `NUM_SM` in host code.** Query it: `cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev)`. In the fork, `NUM_SM=148` is B300-specific; moving it to B200 will under-fill the device.
3. **If cluster / DSMEM / multicast can degrade, the fallback path must be optional** — the CLC preferred+fallback design exists precisely for this.

## References

- [NVIDIA/CUTLASS](https://github.com/NVIDIA/cutlass)
- [djmmoss/ThunderKittens fork](https://github.com/djmmoss/ThunderKittens)
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
