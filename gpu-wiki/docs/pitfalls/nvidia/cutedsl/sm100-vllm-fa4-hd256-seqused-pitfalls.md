# vLLM FlashAttention-4 (CuTeDSL) head_dim=256 + version-selection pitfalls on sm_100

_Last Updated: 2026-07-03_

Traps hit while integrating and validating the vLLM FlashAttention-4 (FA4)
CuTeDSL path on NVIDIA Blackwell (B200 / B300, sm_100 / sm_100a / sm_103). The
FA4 kernel is not in the vLLM main tree — it is pulled at build time from vLLM's
own flash-attention fork at pin commit `2c839c33`; see the integration
reference for provenance:
[sm100-vllm-fa4-integration.md](../../../ref-docs/nvidia/cutedsl/sm100/sm100-vllm-fa4-integration.md).

The two traps below are the reason "the card is Blackwell and FA4 supports
varlen + seqused" is *not* enough to conclude that a given attention call
actually runs on FA4 with seqused.

## 1. head_dim=256 on the dedicated 2-CTA kernel cannot take `seqused_q`/`seqused_k`

### symptom

An attention forward with `head_dim=256`, paged-KV, and per-request valid
lengths (`seqused_q` / `seqused_k`) — the typical vLLM decode requirement —
does not run on FA4. In the pin-commit code the request is not silently
mishandled; it is hard-asserted at `flash_attn/cute/interface.py:978-979`:

```
assert seqused_q is None and seqused_k is None
# "SM100 forward with head_dim=256 does not support seqused_q/seqused_k"
```

The same head_dim=256 code path also disables `softcap`, block sparsity, and
learnable sink.

### cause

On sm_100 / sm_101 Blackwell, `head_dim=256` must go through a *dedicated 2-CTA
forward kernel* because a single CTA's TMEM capacity is insufficient for the
256-wide tile. That specialized 2-CTA kernel never implemented `seqused`.

Upstream chose to assert rather than run without `seqused` because the failure
mode is numerical, not just a missing feature: without valid-length masking,
the uninitialized padding of paged-KV blocks feeds `0 * NaN` into the `P@V`
accumulation (the FA4 hd256 seqused padding-NaN bug). A hard assert surfaces the
constraint instead of returning silently corrupt output.

### fix

On Blackwell, do not expect the official FA4 path to serve `head_dim=256`
paged-attention when you need `seqused` / variable valid lengths. Either:

- fall back to FA2 (this is exactly why vLLM's dispatcher downgrades sm_100
  `head_size > 128 and != 192` to FA2 — see trap 2), or
- write a custom hd256 kernel that explicitly zeroes / masks the paged-KV
  padding.

Do not read "FA4 supports varlen + seqused" as universal: that statement holds
for the general varlen path but is explicitly false on the hd256 2-CTA kernel.

## 2. "The card is Blackwell" does not mean the call runs on FA4

### symptom

Attention perf or numerics do not match FA4 expectations on a B200 / B300, even
though the hardware supports FA4. The call has silently fallen back to FA2 (or
is on FA3), so tuning against "FA4 behavior" chases the wrong kernel.

### cause

vLLM's v1 attention backend selects the FA version in
`vllm/v1/attention/backends/fa_utils.py:get_flash_attn_version()`. The defaults
are:

- sm_90 (Hopper) → FA3
- sm_100 (`device_capability.major == 10` **and** `is_fa_version_supported(4)`)
  → FA4
- everything else → FA2

Even on an FA4-capable sm_100 device, several hard-coded conditions force a
silent downgrade from FA4 back to FA2:

1. `head_size > 128 and head_size != 192` — TMEM capacity limit
   (Dao-AILab issue #1959); this is what excludes `head_dim=256` (trap 1).
2. `VLLM_BATCH_INVARIANT` enabled — FA4's batch-shape scheduling breaks batch
   invariance.
3. ALiBi.
4. MLA with a non-standard paged layout — unsupported by the FA4 CuTeDSL kernel
   (see the comment near `fa_utils.py:299`).

### fix

Before debugging attention perf or numerics on Blackwell, confirm which version
`get_flash_attn_version()` actually returns for the current shape + config.
`head_dim` (especially 256), the batch-invariant switch, ALiBi, or an MLA layout
each independently trigger the FA4 → FA2 fallback. Assume nothing from the arch
alone.

## evidence + reproduction

- Trap 1: the hard assert lives at `flash_attn/cute/interface.py:978-979` of the
  vLLM flash-attention fork pin commit
  `2c839c33742309ec41e620bf837495ec9926c56e`
  (`assert seqused_q is None and seqused_k is None`, error string
  "SM100 forward with head_dim=256 does not support seqused_q/seqused_k"), on
  the sm_100 dedicated 2-CTA `head_dim=256` forward kernel with bf16 / fp8,
  varlen + paged-KV.
- Trap 2: the dispatch and downgrade rules are in
  `vllm/v1/attention/backends/fa_utils.py:get_flash_attn_version()` (vLLM tree
  HEAD `93d8f834d`), spanning sm_90 (H100 / H200) through sm_100 (B200 / B300).

## affected versions

- vLLM flash-attention fork (CuTeDSL FA4) pin commit `2c839c33742309ec41e620bf837495ec9926c56e`
- vLLM tree HEAD `93d8f834d` (`fa_utils.py` dispatch logic)
- Blackwell sm_100 / sm_100a / sm_103 (B200, B300); FA version selection also
  covers sm_90 / sm_90a (H100, H200)
