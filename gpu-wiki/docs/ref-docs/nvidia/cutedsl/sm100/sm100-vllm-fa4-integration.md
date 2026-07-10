# vLLM FlashAttention-4 (CuTeDSL) integration on sm_100 — provenance, varlen/seqused, version selection

_Last Updated: 2026-07-03_

Reference (not a perf journey) for how the FlashAttention-4 (FA4) CuTeDSL kernel
is wired into vLLM on NVIDIA Blackwell, what varlen / seqused surface it exposes,
and when vLLM actually selects it.

| Item | Value |
|---|---|
| Operator | flash_attention (prefill + decode) |
| Hardware | Blackwell B200 / B300 (sm_100 / sm_100a / sm_103); version selection also spans Hopper H100 / H200 (sm_90) |
| DSL | FlashAttention CuTeDSL, pulled into vLLM at build time |
| vLLM tree HEAD (dispatch logic) | `93d8f834d` |
| FA4 fork pin commit (kernel source) | `2c839c33742309ec41e620bf837495ec9926c56e` |

## 1. Provenance: where the FA4 CuTeDSL kernel actually lives

Grepping the `vllm-project/vllm` source tree for the FA4 kernel finds nothing —
and that is expected. The FA4 CuTeDSL implementation (`flash_attn/cute/`) is
**not** in the vLLM main tree. It is an external CuTeDSL component installed at
build time:

- Pulled by CMake from `cmake/external_projects/vllm_flash_attn.cmake`.
- Source repo is `https://github.com/vllm-project/flash-attention.git` — vLLM's
  **own fork**, not the Dao-AILab upstream.
- Pinned at commit `2c839c33742309ec41e620bf837495ec9926c56e`.
- Installed as the CMake component `_vllm_fa4_cutedsl_C` into
  `vllm/vllm_flash_attn/cute/`.
- Runtime import path is `vllm.vllm_flash_attn`. In an environment where the
  component was not built, importing it raises `ModuleNotFoundError`.

Practical consequences:

- To read or modify vLLM's FA4 kernel, go to the fork's pin commit
  (`2c839c33`) — **not** the Dao-AILab upstream, and **not** the vLLM main tree.
- To verify which version is actually loaded at runtime, inspect
  `vllm.vllm_flash_attn` in the live environment.
- A "can't find the kernel in the tree" result is normal, not a broken checkout.

## 2. varlen + seqused support

varlen + paged-KV is the **standard** (not optional) working mode of FA4 inside
vLLM — every attention call goes through `flash_attn_varlen_func`. `seqused` is a
first-class varlen parameter that is explicitly forwarded to the CuTeDSL kernel
and shape-checked, not merely present in the API.

In the pin-commit `flash_attn/cute/interface.py`:

| Element | Location | Detail |
|---|---|---|
| Public varlen entry | `interface.py:2353` | `flash_attn_varlen_func` |
| Underlying forward | `interface.py:360` | `_flash_attn_fwd` |
| Signature args | `interface.py:365-368` | `cu_seqlens_q/k`, `seqused_q/k` all present |
| Validation | `interface.py:461-477` | `seqused_*` must be `(batch_size,)` int32 |
| Internal varlen flag | `interface.py:703-707` | `is_varlen` = any of `cu_seqlens_q/k` or `seqused_q/k` non-empty |

On the vLLM side, the `fa_version == 4` branch in `flash_attn_interface.py`
calls `_flash_attn_fwd(..., seqused_k=seqused_k)`, and the backend uses
`flash_attn_varlen_func` throughout.

The varlen + paged-KV offset convention (useful when porting or aligning against
FA4) is, at `flash_attn.py:1236`:

```
abs_q = q_idx + (seqlen_k - seqlen_q)
```

**Exception:** this general varlen + seqused support does **not** hold on the
`head_dim=256` 2-CTA dedicated kernel, which hard-asserts `seqused_q`/`seqused_k`
are `None`. See
[sm100-vllm-fa4-hd256-seqused-pitfalls.md](../../../../pitfalls/nvidia/cutedsl/sm100-vllm-fa4-hd256-seqused-pitfalls.md)
and the FA4 2-CTA warp-specialization background in
[flash-attention-4-warp-specialization-2cta.md](../../common/sm100/flash-attention-4-warp-specialization-2cta.md).

## 3. Version selection and FA4 → FA2 fallback matrix

vLLM's v1 attention backend picks the FA version in
`vllm/v1/attention/backends/fa_utils.py:get_flash_attn_version()` (vLLM tree
HEAD `93d8f834d`). "Hardware supports FA4" is not the same as "this call runs
FA4."

Default selection by arch:

| Arch | Condition | Selected version |
|---|---|---|
| Hopper (sm_90) | — | FA3 |
| Blackwell (sm_100) | `device_capability.major == 10` **and** `is_fa_version_supported(4)` | FA4 |
| Everything else | — | FA2 |

Hard-coded conditions that force FA4 → FA2 on an otherwise FA4-capable sm_100
device:

| Condition | Reason |
|---|---|
| `head_size > 128 and head_size != 192` | TMEM capacity limit (Dao-AILab issue #1959); this excludes `head_dim=256` |
| `VLLM_BATCH_INVARIANT` enabled | FA4's batch-shape scheduling breaks batch invariance |
| ALiBi | not supported on FA4 |
| MLA with non-standard paged layout | unsupported by FA4 CuTeDSL (comment near `fa_utils.py:299`) |

Rule of thumb: on Blackwell, `head_dim` (especially 256), the batch-invariant
switch, ALiBi, or an MLA layout will each silently drop you to FA2. Confirm the
actual return of `get_flash_attn_version()` for the current shape + config
before attributing any perf or numeric behavior to FA4.

## Related

- Pitfalls distilled from this integration:
  [sm100-vllm-fa4-hd256-seqused-pitfalls.md](../../../../pitfalls/nvidia/cutedsl/sm100-vllm-fa4-hd256-seqused-pitfalls.md)
- FA4 2-CTA warp specialization on Blackwell:
  [flash-attention-4-warp-specialization-2cta.md](../../common/sm100/flash-attention-4-warp-specialization-2cta.md)
