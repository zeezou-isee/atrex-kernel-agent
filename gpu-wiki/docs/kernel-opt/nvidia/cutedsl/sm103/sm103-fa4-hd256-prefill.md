# SM103 FA4 hd256 Prefill: Serving-Correct 1-CTA Path + Dense 2-CTA Tuning (kernel-opt)

> This is an **optimization highlights quick reference** for FlashAttention-4 (FA4) `hd256` prefill in CuTeDSL on B300 (Blackwell Ultra, sm_103). See "Further Reading" for the full v0→v2 journey and the 5 pitfalls that shaped it.

## Trigger Conditions

You are writing or integrating the FA4 CuTeDSL `hd256` forward (`flash_attn.cute`) on **B300 / sm_103**, with:
- head_dim = **256**, GQA (e.g. nqh=32 / nkv=2, 16:1), bf16 in/out + fp32 softmax/accum, causal.
- Either a **dense** benchmark path or a **paged + varlen serving** path (vLLM).

Symptoms this reference addresses:
- The dedicated `hd256` **2-CTA** kernel produces **garbage / all-token-0** output when run inside a real vLLM process alongside other TMEM users (NVFP4 MoE, GDN cutlass) — see pitfalls.
- The dense 2-CTA kernel is **compute-bound but stuck at 57–64% of TFLOPS peak** (ncu Compute SOL ~75%).
- The serving path is stuck at ~60% Compute SOL vs trtllm-gen's ~79% at the same occupancy.

## 1. Drop hd256 to 1-CTA — the fix that removes the garbage

The dedicated `hd256` 2-CTA kernel corrupts its accumulator under concurrent TMEM users (root cause in pitfalls #2). The correct **and** serving-faster path is to run `hd256` through the general `FlashAttentionForwardSm100` in **1-CTA (`CtaGroup.ONE`)** mode, gated by an env flag so vLLM defaults are untouched:

```python
# FA4_HD256_1CTA=1  ->
use_dedicated_hd256_kernel = False
use_2cta_instrs           = False   # CtaGroup.ONE
tile_n                    = 128
q_stage                   = 1       # bf16 MUST use q_stage=1 (see note)
```

- 1-CTA means each CTA computes a full tile in its **own** TMEM: no peer-CTA half, no cross-CTA accumulator, no DSMEM coherence → **immune to concurrent TMEM corruption**.
- **bf16 forces `q_stage=1`**: `q_stage=2` needs 256 KB of SMEM for Q+O alone, over the 224 KB limit.
- Correctness: `test_seqused_paged.py` 14/14 PASS (`PAD_FILL=garbage/zero`, max|diff| ~1e-2 in bf16). Real pai-vllm qwen3.7-max TP4 end-to-end: 3 prompts all emit coherent structured tokens; garbage gone.
- **Serving is also faster than 2-CTA**: decode 4.2–6.4×, prefill 1.5–2.3×. Only a large dense single-sequence prefill keeps 2-CTA ahead (1.1–1.3×), because decode M=1 wastes a cluster and 2-CTA varlen falls into the slow `SingleTileVarlenScheduler`.

## 2. tile_n = 96 to unlock kv_stage=2 (shipped serving win)

On the 1-CTA path, `kv_stage=1` at `tile_n=128` costs 199 KB SMEM and leaves no KV prefetch to overlap the per-KV-block softmax wait. Lowering `tile_n` **halves per-stage KV-SMEM**, which fits `kv_stage=2` prefetch plus TMEM headroom:

```python
# FA4_TILE_N=96  (later made the 1-CTA default)
```

- `tile_n < 128` and `!= page_size` **drops out of TMA** and uses a non-TMA paged gather (cp.async), lowering L1 hit — but the net effect is positive.
- `tile_n=96` is the **largest** tile that still admits `kv_stage=2`: `112` falls back to `kv_stage=1`, `64` forces `kv_stage=3` (too many iterations, regresses to 1052–1066 TF).
- 4 shapes: 12–18% faster (0.848 / 0.928 / 1.014 / 1.032 ms vs 0.99 / 1.08 / 1.15 / 1.21 ms), 1119–1167 TF.
- ncu @ S7680: Compute SOL 46.8→59.6%, tensor active 44.5→53.9%, barrier stall 7.31→4.08, long-scoreboard 8.93→6.19, occupancy 15.5→19.9% (now above trtllm's 18.6%), dyn SMEM 198→231 KB.
- Numerics identical to `tile_n=128` (14/14 PASS). Real TP4 A/B: 12.03 vs 13.04 µs/call (~8% faster), output tokens byte-identical.

## 3. exp2 interleave frequency (dense 2-CTA, ~3% lever)

The `hd256`-causal software exp2 emulation (Horner polynomial) runs on the softmax warp's FMA units and contends with the correction warp. Spacing the interleave out gives MMA a more continuous execution window:

```python
# _TUNING_CONFIG, hd256 causal:  ex2_emu_freq 14 -> 20
```

- V0→V1 per shape: S7680 −0.3%, S8064 −2.5%, S8320 −0.8%, **S8576 −8.2%** (largest), mean TFLOPS 1324→1364 (+3%), rel_l2 unchanged.
- This is a **safe ~3% knob**, not a structural fix: SM Throughput is already 74.98% while only 57–64% of TFLOPS peak, because only 1 of 12 warps issues MMA, the software-exp2 FMA is not counted in the FLOP budget, and 39.1% of warp stall is the TMA↔MMA producer/consumer bubble.

## 4. SM103 hardware SFU exp2 (dense 2-CTA, arch-specific)

The software exp2 emulation exists to route around SM100's slow SFU. On **SM103 the SFU is faster**, so the emulation is pure overhead. FA4's non-hd256 kernels already switch this off, but the dedicated `hd256` kernel hard-coded `is_sm103=False` and never got it. Add the arch check and disable emulation:

```python
# BaseDSL._get_dsl().get_arch_enum() -> is_family_of(sm_103f)
# on SM103:  ex2_emu_freq = 0   -> use hardware SFU exp2
```

- A/B (same seed=42, warmup=50, rep=200): **S8320 +15.5% (926→782 µs, 1225→1450 TF)**, **S8576 +14.3% (961→824 µs, 1254→1463 TF)**, S7680 unchanged (exp2 share negligible on short shapes), S8064 −1.3% (noise). rel_l2 ~0.002 unchanged.
- Gains scale with softmax's share of the work, so they concentrate on the long shapes (S≥8320).

## 5. Build vLLM + FA4 into an existing env

Editable-installing vllm + fa4 (`vllm_flash_attn`) into conda env `xingze` (py3.12, torch 2.11.0+cu130, nvcc 13.2) on B300, with build CPU cgroup-limited to ~1 core:

- vllm: run `use_existing_torch.py` to strip torch pins, then `pip install -e . --no-build-isolation`.
- fa4: `pip install -e . --no-build-isolation --no-deps` — **`--no-deps` is mandatory**; without it, FA4 metadata pins `torch==2.4.0` (797 MB) and overwrites torch 2.11.
- vllm needs `setuptools>=77` (env had 70.2.0; PEP-639 `license="Apache-2.0"` fails) → upgraded to 80.10.2.
- Build times under 1-core quota: vllm ~50 min, fa4 ~3 h (MAX_JOBS is useless here — budget 3 h+).
- `test_fa4_smoke` still fails afterward: the bundled `flash_attn/__init__.py` hard-imports legacy `flash_attn_2_cuda` (only present in PAI wheels, not built here). Wrap that import in `try/except` to use `flash_attn.cute`.

## 6. Wire hd256 into vLLM's varlen / paged path (seqused_k)

vLLM's FA4 wrapper always passes `seqused_k` (`fa_utils.py:145`), but the baseline `hd256` 2-CTA kernel hard-asserts against it → warmup crashes with `SM100 forward with head_dim=256 does not support seqused_q/seqused_k`. Two paths:

- **Prefill-only workaround**: remove `seqused_k` from the wrapper signature so `has_seqused_k=False` (the flag is probed via `inspect.signature`); vLLM then passes `cu_seqlens_k`, which `hd256` supports and which fully describes the KV for `max_tokens=1` prefill.
- **Real fix** (branch `fa4_bf16_varlen`, commit `2b1511f`): add `seqused_q/k` to the `hd256` kernel — kernel param plus four warp-segment per-batch overrides of `seqlen_k = seqused_k[batch_coord]`, flowing through `get_trip_start_count_via_block_info` to truly truncate the KV loop. Forward rel_l2 ≈ 0 (zero regression); backward still asserts (forward only).
- Requirements for the hd256 paged path: **`page_size == 128` (TMA-only)** and a **row-major contiguous `block_table`**. Decode (varlen encoded only by `seqused_k`) cannot run without the real fix; only forward prefill can bypass via `cu_seqlens_k`.

## Measured Benefits

**Serving 1-CTA path** (paged + varlen, S ∈ {7680, 8064, 8320, 8576}):

| Change | Metric | Before | After |
|---|---|---|---|
| 2-CTA → 1-CTA (§1) | correctness in vLLM | garbage / all-token-0 | 14/14 PASS, coherent tokens |
| 2-CTA → 1-CTA (§1) | decode latency | — | **4.2–6.4× faster** than 2-CTA |
| 2-CTA → 1-CTA (§1) | prefill latency | — | **1.5–2.3× faster** than 2-CTA |
| tile_n 128 → 96 (§2) | latency (4 shapes) | 0.99 / 1.08 / 1.15 / 1.21 ms | **0.848 / 0.928 / 1.014 / 1.032 ms** (12–18%) |
| tile_n 128 → 96 (§2) | Compute SOL @ S7680 | 46.8% | **59.6%** |
| tile_n 128 → 96 (§2) | real TP4 per-call | 13.04 µs | **12.03 µs** (~8%) |

**Dense 2-CTA path** (benchmark reference, not serving-usable):

| Change | Shape | Before | After |
|---|---|---|---|
| exp2 freq 14→20 (§3) | S8576 | 946 µs | **868 µs** (−8.2%) |
| exp2 freq 14→20 (§3) | mean TFLOPS | 1324 | 1364 (+3%) |
| SM103 hw exp2 (§4) | S8320 | 926 µs (1225 TF) | **782 µs (1450 TF)** (+15.5%) |
| SM103 hw exp2 (§4) | S8576 | 961 µs (1254 TF) | **824 µs (1463 TF)** (+14.3%) |

## Anti-Patterns

| Don't Try | Because |
|---|---|
| Run dedicated `hd256` 2-CTA in a concurrent vLLM process | Known TMEM-capacity bug (Dao-AILab #1959): accumulator overflows 512 TMEM columns → NaN on 49 fixed even channels → garbage. Use 1-CTA (§1). |
| "Fix" the 2-CTA NaN with a V-SMEM flush | Removes the 19000 NaNs but leaves accumulator corruption (max|out−ref| ~2e38); output still garbage. |
| Chase the 1-CTA gap with CLC persistent + `split_P_arrive` sweeps | B=1 prefill is already balanced; both give 0.988–0.998 ms (null). The real limiter is the softmax-wait barrier stall. |
| Set `kv_stage=3` (dense 2-CTA) | Latency-hiding starves → catastrophic −40% (drops to ~40% peak). Sweet spot is `kv_stage=4`. |
| Set `kv_stage=5` chasing deeper pipeline | Extra SMEM buys no occupancy (already pinned at 18.75%) → ~56%, slightly worse. |
| Single-warpgroup S-buffer ping-pong overlap in CuTeDSL 4.5.2 | In-process MLIR/LLVM compile explodes (7.5–29 min, killed). Cross-buffer commit/acquire is not legalizable; 4.6.0 hits the same wall. |
| `ex2_emu_res=3` (lower-order polynomial) | No improvement, slightly worse. |
| `pip install -e .` FA4 without `--no-deps` | Pulls `torch==2.4.0` and overwrites your CUDA stack. |

## Further Reading

- **Pitfalls (5 traps, symptom → cause → fix)**:
  [`docs/pitfalls/nvidia/cutedsl/sm103-fa4-hd256-prefill-pitfalls.md`](../../../../pitfalls/nvidia/cutedsl/sm103-fa4-hd256-prefill-pitfalls.md)
- **Full optimization journey (v0 dense → v1 → v2 + trtllm-gen ceiling)**:
  [`docs/ref-docs/nvidia/cutedsl/sm103/sm103-fa4-hd256-prefill-optimization.md`](../../../../ref-docs/nvidia/cutedsl/sm103/sm103-fa4-hd256-prefill-optimization.md)
