# FlashAttention-4 on sm_103a (B300) — Arch Dispatch

How FlashAttention-4 (FA4, the CuTeDSL implementation) targets NVIDIA B300 / sm_103a (Blackwell Ultra, compute capability 10.3): there is no dedicated `sm103` build target — every 10.x device is routed into the SM100 dispatch bucket, so FA4 on B300 runs the sm_100 compiled artifacts, including the hd256 2-CTA kernel and its known TMEM-contention bug.

## Provenance

- Upstream: <https://github.com/Dao-AILab/flash-attention>
- vLLM fork: <https://github.com/vllm-project/flash-attention> (the one vLLM builds use; PRs are vetted here first)
- Main directory: `flash_attn/cute/` (FA4, CuTeDSL implementation)
- Install: `pip install flash-attn-4` (add the `[cu13]` extra on CUDA 13)
- User entry point: `from flash_attn.cute import flash_attn_func`
- Related local references (paths omitted): upstream Dao-AILab checkout, vllm-project fork checkout, and an internal pai-vllm fork (URL omitted) that embeds a copy of CUTLASS (`cluster_sm100` / `tmem_allocator_sm100`).
- Pull / research date: 2026-07-03

## Blackwell Operators Covered

The arch targets already shipped in `flash_attn/cute/`:

- `sm90` — Hopper (H100)
- `sm100` — datacenter Blackwell (B200)
- `sm120` — consumer Blackwell (RTX Blackwell)

**There is no sm103 file.** `interface.py._parse_arch_str` puts every 10.x device into the SM100 dispatch bucket:

```python
# arch // 10 == 10 → SM100 bucket
```

`sm100_hd256_2cta_fmha_forward.py` carries an inline comment:

```python
# hd256: always 2cta, no sm103 variant
```

So on B300 (sm_103a), FA4 runs the sm_100 compiled artifacts; there is no sm_103a-specific tuned build. See [sm103-vs-sm100-differences.md](sm103-vs-sm100-differences.md).

## sm_100 files (reused on sm_103a)

| File | Coverage |
|---|---|
| `sm100_hd256_2cta_fmha_forward.py` | hd256 forward, **must be 2-CTA**; a single CTA overflows TMEM |
| `sm100_hd256_2cta_fmha_backward.py`, `_backward_dkdvkernel.py`, `_backward_dqkernel.py` | hd256 backward, split into two kernels (one for dK/dV, one for dQ) |
| `flash_fwd_mla_sm100.py`, `flash_bwd_mla_sm100.py` | MLA (Dqk=192, Dvo=128) |
| `blackwell_helpers.py` | shared Blackwell helpers (TMEM offset management, etc.) |
| `mma_sm100_desc.py` | tcgen05.mma descriptor |
| `interface.py` | user-facing API + arch dispatch |

## The dispatch guard you must know on sm_103a

**The head_size guard upstream in vLLM** (`vllm/v1/attention/backends/fa_utils.py`):

```python
# FA4 on SM100 (Blackwell) has TMEM capacity limits that restrict supported
# head dimensions.
# See: https://github.com/Dao-AILab/flash-attention/issues/1959
# Exception: hdim 192 is supported for MLA's diff-headdim case (qk=192, v=128)
if (fa_version == 4 and device_capability.major >= 10
        and head_size is not None and head_size > 128 and head_size != 192):
    logger.warning_once(
        "FA4 on Blackwell does not support head_size=%d due to TMEM "
        "capacity limits, defaulting to FA version 2.", head_size)
    fa_version = 2
```

This means **on sm_103a, when head_dim > 128 and ≠ 192, vLLM automatically downgrades FA4 back to FA2**. The internal pai-vllm fork (commit `b46dc08`) lacks this guard, so on a model with head_dim=256 it hit garbage output — the root-cause / fix history is captured in a local snapshot (path omitted). See also the vLLM dispatch reference [vllm-sm103-attention-dispatch.md](vllm-sm103-attention-dispatch.md).

## Known hard bug on sm_103a: hd256 TMEM contention

**Symptom:** internal pai-vllm (missing the guard) + FA4 + head_dim=256 → huge values / NaN in the output → all-token-0 garbage. Offline unit tests pass; it only breaks in a real forward pass when running concurrently with other kernels (NVFP4 MoE, GDN).

**Cause:** the hd256 kernel uses `tmem_o_offset=256`, so the S accumulator occupies col [0, 256) and the O accumulator occupies col [256, 512) — **a single kernel fills the entire 512-column TMEM block**. In a real forward pass, NVFP4 MoE and GDN CUTLASS kernels are also requesting TMEM at the same time; after the conflict the accumulators read corrupted data → huge values / NaN.

**Fix:** take the head_size guard and fall back to FA2 — i.e. the upstream approach (Dao-AILab issue #1959). Do not try to hard-patch the hd256 TMEM budget — upstream has declared it "unsupported".

See the pitfalls write-up [sm100-vllm-fa4-hd256-seqused-pitfalls.md](../../../../pitfalls/nvidia/cutedsl/sm100-vllm-fa4-hd256-seqused-pitfalls.md) and [blackwell-tmem-tensor-memory.md](blackwell-tmem-tensor-memory.md).

## FP8 / varlen / seqused_k side: pai-vllm patch

- Commit `ed145efcc0400274d25a44e70b9473c327eb8b1e8` (internal pai-vllm fork) added FA4 fp8 varlen optimizations, including an hd256 branch — extracted to a local snapshot (path omitted, pure snapshot with no `.git`).
- The `varlen` parameter (cu_seqlens packing + single batch) is used only in prefill; decode goes through paged KV (`block_table` + `seqused_k`).
- `seqused_k` affects the kernel: it is the actual per-batch upper bound on KV length, deciding the number of K/V loop iterations, the page-walk depth, and the KV cache load volume. It does **not** affect the hd256 TMEM bug — that is a pure accumulator-column-occupancy issue, independent of K/V length.

## Status of upstream fmhaSm103

From a captured kernel trace (local snapshot, path omitted), the kernel names seen:

- `flash_attn_cute_sm100_hd256_2cta_fmha_forward` — the Dao-AILab/flash-attention sm100 hd256 2-CTA forward (via CuTeDSL JIT)
- **not** `fmhaSm103aKernel_...` (that is a trtllm-gen cubin, only reached when vLLM uses the FLASHINFER backend)

trtllm-gen's `fmhaSm103aKernel_...H256PagedKvCausal...` does have a genuine sm_103a-only cubin, but on pure upstream vLLM 0.23 + hd256 it fails to load (see the A/B comparison experiments recorded for the internal pai-vllm fork) — not recommended to rely on in the short term. See the baseline pitfalls [sm103-trtllm-gen-paged-decode-baseline-pitfalls.md](../../../../pitfalls/nvidia/cuda/sm103-trtllm-gen-paged-decode-baseline-pitfalls.md).

## References

- README (install and API): <https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/README.md>
- PyPI: <https://pypi.org/pypi/flash-attn-4/json> (v4.0.0b20, declares the `cu13` extra)
- hd256 TMEM bug upstream issue: <https://github.com/Dao-AILab/flash-attention/issues/1959>
- vLLM-side fa_utils.py guard: <https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/fa_utils.py>
- Sibling docs: [vllm-sm103-attention-dispatch.md](vllm-sm103-attention-dispatch.md), [sm103-vs-sm100-differences.md](sm103-vs-sm100-differences.md), [blackwell-tmem-tensor-memory.md](blackwell-tmem-tensor-memory.md)
