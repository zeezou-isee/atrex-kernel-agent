# vLLM — sm_103a Attention Backend Dispatch

How vLLM targets NVIDIA B300 / sm_103a (Blackwell Ultra, compute capability 10.3): vLLM ships no sm_103a-specific kernel, but its attention-backend dispatch is the first critical point where things go wrong on sm_103a — the `fa_utils.py` head_size guard is what keeps a Blackwell card from running the buggy FA4 hd256 path.

## Provenance

- Upstream: <https://github.com/vllm-project/vllm> (0.23 already contains the hd256 guard on the sm_103a side)
- Internal pai-vllm fork (URL omitted): commit `b46dc08` is missing the guard; see the fa4-hd256 incident record.
- FA4-fp8 patch: commit `ed145efc6400274d25a44e70b9473c327eb8b1e8`, extracted to a local snapshot (path omitted, no `.git`).
- Pull / research date: 2026-07-03

## Kernel-side changes relevant to sm_103a

vLLM itself contains no sm_103a-specific kernel, but **its attention backend dispatch is the first critical point that breaks on sm_103a**.

### The hd256 guard in `fa_utils.py` (upstream 0.23)

`vllm/v1/attention/backends/fa_utils.py`:

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

- SM cc ≥ 10 (i.e. both sm_100a and sm_103a count as Blackwell)
- head_size > 128 and ≠ 192 → fall back to FA2
- Exception: 192 (the opening left for MLA's Dqk=192, Dvo=128 shape)

**The internal pai-vllm fork (`b46dc08`) is missing this block**, so 15 hd256 full-attention layers of a head_dim=256 model were forced onto FA4, hit the hd256 TMEM-contention bug, and produced all-token-0 garbage. Fix: copy this upstream block into the internal fork. See the pitfalls write-up [sm100-vllm-fa4-hd256-seqused-pitfalls.md](../../../../pitfalls/nvidia/cutedsl/sm100-vllm-fa4-hd256-seqused-pitfalls.md) and the FA4 dispatch reference [flash-attention-4-sm103-dispatch.md](flash-attention-4-sm103-dispatch.md).

### FA version dispatch (overview)

In `fa_utils.py` (simplified):

- SM90 → FA3
- SM100 (including sm_100a, sm_103a) → FA4
- everything else → FA2

The SM100 branch additionally carries head_size / dtype guards.

## FA4-fp8 patch (commit `ed145efc`)

The internal pai-vllm fork added FA4 fp8 optimizations + varlen support, including an hd256 branch. It has been extracted as a pure snapshot (6001 files / 112 M). Key points:

- `VLLM_FLASH_ATTN_SRC_DIR` can point at a local flash-attn source directory, for easier iteration.
- The FA4 backend goes through `flash_attn.cute.flash_attn_func`.
- It does **not** add the hd256 guard above.

## sm_103a attention kernel names seen in traces (for cross-checking)

From a captured trace (local snapshot, path omitted):

- `flash_attn_cute_sm100_hd256_2cta_fmha_forward` — Dao-AILab FA4 sm100 hd256 2-CTA forward (reached on sm_103a via the sm100 dispatch)
- `cross_device_reduce_1stage` — vLLM's built-in custom AllReduce (replacing NCCL)

## MoE grouped-GEMM side (the actual kernels on sm_103a)

Seen from a prefill trace:

- `bmm_E2m1`, `bmm_Bfloat16_E2m1E2m1` — NVFP4 grouped-GEMM (E2M1 weights)
- `compute_problem_sizes` — per-expert token count
- `expandInputRowsKernel`, `doActivationKernel`, `finalizeMoeRouting` — MoE routing / activation / combine

These are all CUTLASS or trtllm-gen kernels underneath; vLLM only integrates them — sm_103a tuning belongs to the CUTLASS arch layer, see [cutlass-blackwell-sm103-arch-layer.md](cutlass-blackwell-sm103-arch-layer.md) and the trtllm-gen baseline pitfalls [sm103-trtllm-gen-paged-decode-baseline-pitfalls.md](../../../../pitfalls/nvidia/cuda/sm103-trtllm-gen-paged-decode-baseline-pitfalls.md).

## Known sm_103a issues (pai-vllm)

- hd256 garbage: root cause identified. The fix is the upstream guard.

## References

- fa_utils guard: <https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/fa_utils.py>
- hd256 upstream bug: <https://github.com/Dao-AILab/flash-attention/issues/1959>
- vLLM FA4 integration commit `8b5014d3`: <https://github.com/vllm-project/vllm/commit/8b5014d3dd>
- Sibling docs: [flash-attention-4-sm103-dispatch.md](flash-attention-4-sm103-dispatch.md), [cutlass-blackwell-sm103-arch-layer.md](cutlass-blackwell-sm103-arch-layer.md)
