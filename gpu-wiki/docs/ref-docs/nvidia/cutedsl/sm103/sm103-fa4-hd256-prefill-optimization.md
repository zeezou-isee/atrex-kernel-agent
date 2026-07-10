# FA4 hd256 Prefill on B300 (sm_103) — Optimization Journey

End-to-end journey of optimizing the FlashAttention-4 (FA4) `hd256` prefill
kernel in CuTeDSL on **B300 (Blackwell Ultra, sm_103)**, across two tracks: a
**dense benchmark** track (v0→v1→v2 on the dedicated 2-CTA kernel) that chased
the compute-bound ceiling, and a **serving** track (paged + varlen in vLLM) that
had to first make the kernel *correct* before making it fast. It ends by
adopting trtllm-gen as the reachable ceiling and keeping the FA4 1-CTA kernel as
a correctness fallback.

**Last Updated**: 2026-07-03 · **Attempts**: 3 dense versions + 5 serving/tuning experiments + 1 ceiling comparison

## Target hardware & config

| Item | Value |
|---|---|
| GPU | B300 (Blackwell Ultra), sm_103 (reports as L20D cc8.9; real cc=(10,3)) |
| DSL | CuTeDSL / nvidia-cutlass-dsl 4.5.2 (ping-pong also tried on 4.6.0) |
| Kernel | FA4 `flash_attn.cute`: dedicated `sm100_hd256_2cta_fmha_forward` + general `FlashAttentionForwardSm100` |
| dtype | bf16 in/out, fp32 softmax/accum |
| Shape | head_dim=256, GQA nqh=32 / nkv=2 (16:1), causal, S ∈ {7680, 8064, 8320, 8576} |
| TMEM | 512 columns · SMEM 224 KB |
| Roofline | AI = 3614–4038 ≫ ridge = 281 → **strongly compute-bound** |
| 90% peak target | 2025 TFLOPS / 477–595 µs per shape |

## Version ladder

| Ver | Outcome | Technique | Measured (4 shapes S={7680,8064,8320,8576}) |
|---|---|---|---|
| **V0** dense baseline | neutral | dedicated 2-CTA hd256, unmodified (tcgen05.mma, TMEM, TMA, software exp2, warp-spec) | 668 / 829 / 880 / 946 µs; 64.3 / 57.1 / 57.3 / 56.6% peak; 4/4 PASS (rel_l2 0.00199); 3.6–4.1× over bladnn_fa4 paged |
| **V1** dense tune | positive (small) | exp2 interleave freq `ex2_emu_freq` 14→20 | S8576 −8.2% (868 µs); mean TFLOPS 1324→1364 (+3%); rel_l2 unchanged |
| V1 sub: kv_stage | negative | sweep `kv_stage ∈ {3,4,5}`, `ex2_emu_res=3` | kv_stage=4 optimal; **kv_stage=3 −40%**; kv_stage=5 ~56%; res=3 no gain |
| **V2** dense arch | positive | SM103 detect → `ex2_emu_freq=0` (hardware SFU exp2) | S8320 +15.5% (926→782 µs, 1450 TF); S8576 +14.3% (961→824 µs, 1463 TF); S7680 flat |
| Serving: 2-CTA in vLLM | negative | dedicated hd256 2-CTA end-to-end | all-token-0 garbage; kernel writes NaN (`out_nan`≈19000) on 49 fixed even channels |
| Serving: 1-CTA fix | positive | `CtaGroup.ONE`, general kernel, `q_stage=1` | 14/14 PASS, coherent tokens; decode 4.2–6.4×, prefill 1.5–2.3× faster than 2-CTA |
| Serving: cheap levers | negative | CLC persistent, `split_P_arrive` sweep | 0.988–0.998 ms (null); bottleneck = softmax-wait barrier stall |
| **Serving V1 (shipped)**: tile_n=96 | positive | `tile_n` 128→96 → `kv_stage=2` prefetch | 0.848 / 0.928 / 1.014 / 1.032 ms (12–18%); Compute SOL 46.8→59.6% |
| Serving: ping-pong overlap | negative | single-warpgroup S-buffer 2-stage ping-pong | CuTeDSL 4.5.2/4.6.0 compile explosion (7.5–29 min, killed) |
| Ceiling ref: trtllm-gen | neutral | flashinfer 0.6.12 trtllm-gen FMHA cubin | 612–765 µs; Compute SOL 78.8–80.6%; ~1.3–1.5× faster than 1-CTA |

---

## V0 — dense 2-CTA baseline (neutral)

Ran the pip-installed FA4 dense kernel `sm100_hd256_2cta_fmha_forward` (2-CTA
cooperative, tcgen05.mma, TMEM, TMA, software exp2, warp specialization)
unmodified, `do_bench` on 4 production shapes.

- **Measured**: S7680 668 µs (1446 TF, 64.3% peak), S8064 829 µs (1286 TF,
  57.1%), S8320 880 µs (1289 TF, 57.3%), S8576 946 µs (1274 TF, 56.6%);
  rel_l2 max 0.00199, 4/4 PASS. **3.6–4.1× faster** than the production paged
  baseline bladnn_fa4 (2725–3364 µs), but only 57–64% of TFLOPS peak.
- **Analysis**: compute-bound is confirmed (AI ≫ ridge). The 35–40% gap to peak
  comes from the warp-specialized kernel's intrinsic **18.75% occupancy** (12
  warps/CTA, large register + SMEM budget) and producer/consumer pipeline
  bubbles — **not** bandwidth. S7680 is fastest, likely from better tile /
  wave quantization. The dense interface is only a capability upper bound; real
  serving needs paged + varlen (see serving track).

## V1 — dense parameter tuning (positive, small)

Two knobs on the dedicated 2-CTA kernel.

**exp2 interleave frequency** (`_TUNING_CONFIG`, hd256 causal): `ex2_emu_freq`
14→20, spacing the softmax warp's Horner-polynomial software exp2 out relative to
the correction warp's FMAs.
- V0→V1: S7680 −0.3%, S8064 −2.5%, S8320 −0.8%, **S8576 −8.2%**; mean TFLOPS
  1324→1364 (+3%); rel_l2 unchanged.
- ncu: SM Throughput 74.98%, No-Eligible 61.36%, top stall = L1TEX scoreboard
  (39.1% of the 7.8-cyc average warp stall), grid (64,32)×384.
- This is a safe **~3% lever**. The structural ceiling is untouched: only 1 of 12
  warps issues MMA, software-exp2 FMAs are not in the FLOP budget, and 39% of the
  stall is the TMA↔MMA bubble.

**kv_stage sweep** (negative sub-experiment): `kv_stage ∈ {3,4,5}` and
`ex2_emu_res=3`.
- `kv_stage=4` optimal (57–64% peak); **`kv_stage=3` catastrophic −40%** (~40%
  peak, pipeline starves — TMA/MMA cannot overlap); `kv_stage=5` ~56% (extra
  SMEM buys no occupancy, already pinned at 18.75%); `ex2_emu_res=3` no gain.
- Lesson: multi-buffer depth has a sharp sweet spot — too shallow starves (very
  costly), too deep is a no-op when occupancy is saturated.

## V2 — SM103 hardware SFU exp2 (positive)

The dedicated `hd256` kernel hard-coded `is_sm103=False` and always ran the
software exp2 emulation — a workaround for SM100's slow SFU that is pure overhead
on SM103, whose SFU is faster. FA4's non-hd256 kernels already switch this off
(`flash_fwd_sm100.py:205`); the dedicated kernel never got it. Added the arch
check (`BaseDSL...get_arch_enum()`, `is_family_of(sm_103f)`) and set
`ex2_emu_freq=0` to use hardware exp2 on SM103.

- A/B (same seed=42, warmup=50, rep=200): **S8320 926→782 µs (+15.5%, 1225→1450
  TF)**, **S8576 961→824 µs (+14.3%, 1254→1463 TF)**, S7680 unchanged (exp2
  share negligible on short shapes), S8064 −1.3% (noise); rel_l2 ~0.002
  unchanged (hardware exp2 accurate enough).
- Gains scale with softmax's share of the work → concentrated on long shapes.
- Lesson: when porting across generations, a workaround for the old arch's
  bottleneck can become a burden on the new one — gate it by arch. Dedicated
  kernels often lag the general kernel's arch optimizations.

## Serving track — correctness before speed

### 2-CTA in vLLM (negative — the correctness wall)

Integrating the dedicated `hd256` 2-CTA kernel into real pai-vllm (qwen3.7-max,
NVFP4 MoE, TP4, paged + varlen + seqused_k, concurrent NVFP4 MoE / GDN cutlass
kernels) produced **all-token-0 garbage**. A same-process differential probe
proved the **kernel** writes NaN (`out_nan`≈19000 vs `ref_nan`=0) on 49 fixed
even output channels, deterministic per (token,head) and per TP rank — while the
identical binary produced 0 NaN in every offline configuration.

Root cause: known FA4 `hd256` TMEM-capacity bug (Dao-AILab #1959). The
accumulator already fills all 512 TMEM columns (`tmem_s_offset=0` +
`tmem_o_offset=256`); adding `P` + 2-CTA cross-CTA accumulation overflows →
corruption → huge values / NaN. The even channels are the 2SM MMA peer-CTA half,
unreliable under concurrent TMEM users. A both-CTA V-SMEM flush drops `out_nan`
19000→0 but output is still ~2e38 garbage — it masks the NaN, not the overflow.
(Full detail in pitfalls #1.)

### 1-CTA fix (positive — the real fix)

Gated by `FA4_HD256_1CTA=1`, route `hd256` through the general
`FlashAttentionForwardSm100` in 1-CTA (`CtaGroup.ONE`, `use_2cta_instrs=False`,
`tile_n=128`, `q_stage=1`). bf16 **must** use `q_stage=1` (`q_stage=2` needs
256 KB SMEM for Q+O > 224 KB). Each CTA computes a full tile in its own TMEM —
no peer-CTA half, no cross-CTA coherence → immune to concurrent corruption.

- `test_seqused_paged.py` 14/14 PASS (`PAD_FILL=garbage/zero`; the single
  `PAD_FILL=nan` fail is a harness artifact — the general kernel lacks a trailing
  partial-tile NaN flush, but vLLM's KV cache is `torch.zeros`, so 0×0=0, no NaN).
- Real pai-vllm qwen3.7-max TP4: 3 prompts all coherent; garbage gone.
- Serving is also faster: decode **4.2–6.4×**, prefill **1.5–2.3×** vs 2-CTA;
  only large dense single-sequence prefill keeps 2-CTA ahead (1.1–1.3×).

To wire it into vLLM's varlen path, the kernel also needed `seqused_k` support:
prefill-only can bypass via `cu_seqlens_k` (drop `seqused_k` from the wrapper
signature so `has_seqused_k=False`), but decode requires the real fix (branch
`fa4_bf16_varlen`, commit `2b1511f`) that adds per-batch `seqlen_k =
seqused_k[batch_coord]` overrides through `get_trip_start_count_via_block_info`.
Requires `page_size == 128` (TMA-only) and a row-major `block_table`.

### Cheap levers (negative)

CLC persistent scheduling (`FA_CLC=1`) and `split_P_arrive ∈ {32,64,96}` gave
0.988–0.998 ms — null. ncu vs trtllm-gen: Duration 960 vs 612 µs, Compute SOL
46.8 vs 78.8%, occupancy 15.5 vs 18.6% (both low). Smoking gun: `stall_barrier`
7.31 vs 0.002 — with `q_stage=1` there is no second Q-tile to overlap, so MMA
blocks in `producer_acquire` every KV block waiting for softmax to produce `P`.
The limiter is pipeline efficiency, not occupancy.

### Serving V1 (shipped) — tile_n=96 (positive)

Lowering `tile_n` 128→96 halves per-stage KV-SMEM, which fits `kv_stage=2`
prefetch. `tile_n < 128` (and `!= page_size`) drops out of TMA into a cp.async
paged gather (L1 hit drops), but the net effect is positive. `tile_n=96` is the
largest tile that still admits `kv_stage=2` (112 → kv_stage=1; 64 → kv_stage=3,
too many iterations, 1052–1066 TF).

- 4 shapes: 0.848 / 0.928 / 1.014 / 1.032 ms vs 0.99 / 1.08 / 1.15 / 1.21 ms
  (12–18% faster), 1119–1167 TF.
- ncu @ S7680: Compute SOL 46.8→59.6%, tensor active 44.5→53.9%, barrier stall
  7.31→4.08, long-scoreboard 8.93→6.19, occupancy 15.5→19.9% (now above
  trtllm's 18.6%), dyn SMEM 198→231 KB.
- 14/14 PASS, numerics identical; real TP4 12.03 vs 13.04 µs/call (~8%), tokens
  byte-identical. Note: occupancy now exceeds trtllm's yet SOL is still lower —
  confirming occupancy is not the limiter.

### Ping-pong overlap (negative — the ceiling)

Trying to close the remaining SOL gap by 2-stage ping-pong of the single
softmax-warpgroup's `S` buffer (expected +5–7% SOL, 60→~66%). All three coding
strategies exploded the in-process MLIR/LLVM compile (7.5 / 20 / 29 min, killed;
100% CPU, no ptxas). A plain `s_stages=2` sizing compiles fast and PASSes
(rel_l2 1.86e-3). The QK-ahead cross-buffer `commit(nbuf)`/`acquire(buf)`
structure is not legalizable in 4.5.2; 4.6.0 hits the same wall. `tile_n=96` is
this kernel's reachable ceiling under 4.5.2. (Detail in pitfalls #3.)

## Performance-ceiling reference — trtllm-gen

To decide build-vs-adopt, benchmarked the trtllm-gen FMHA shipped in flashinfer
0.6.12 (`trtllm_batch_context_with_kv_cache`, paged block_tables + seq_lens, GQA,
causal), `ncu --set full`, same 4 shapes.

| Path | Latency / Duration | Compute SOL | Serving usable? |
|---|---|---|---|
| **trtllm-gen** (`fmhaSm103a...H256PagedKvCausalP64...PersistentContext`) | 612–765 µs (event 0.654–0.922 ms), ~1300–1478 TF, occ 18.6% | **78.8–80.6%** | yes (paged, native varlen) |
| FA4 dense 2-CTA | 0.69–0.87 ms | 74–76% | no (dense interface only) |
| FA4 1-CTA varlen (tile_n=96) | 0.80–0.99 ms | 60% | yes |
| FA4 varlen 2-CTA | 1.2–1.48 ms | 44% | yes (slow) |

trtllm-gen is both fastest and on the usable paged path — **~1.3–1.5× faster
than FA4 1-CTA**, with SOL even higher than FA4's offline dense. It reaches 79%
SOL at the **same 18.6% occupancy** via persistent-context scheduling + fuller
tcgen05 overlap; FA4 1-CTA is stuck at 60% because CuTeDSL 4.5.2 cannot compile
the ping-pong overlap.

**Decision**: adopt the flashinfer-packaged trtllm-gen native cubin for hd256
paged serving (at ceiling, native varlen, zero kernel risk); keep the FA4 1-CTA
kernel as the correctness fallback. When selecting, benchmark flashinfer's
trtllm-gen first, then decide whether to invest in a custom kernel.

## Related docs

- **Quick reference (positive techniques, apply-in-order)**:
  [`docs/kernel-opt/nvidia/cutedsl/sm103/sm103-fa4-hd256-prefill.md`](../../../../kernel-opt/nvidia/cutedsl/sm103/sm103-fa4-hd256-prefill.md)
- **Pitfalls (5 traps, symptom → cause → fix)**:
  [`docs/pitfalls/nvidia/cutedsl/sm103-fa4-hd256-prefill-pitfalls.md`](../../../../pitfalls/nvidia/cutedsl/sm103-fa4-hd256-prefill-pitfalls.md)
