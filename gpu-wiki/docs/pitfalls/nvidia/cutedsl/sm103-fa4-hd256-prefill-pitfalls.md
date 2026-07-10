# sm_103 FA4 hd256 prefill pitfalls

Traps hit while bringing the FlashAttention-4 (FA4) `hd256` forward
(`flash_attn.cute`, CuTeDSL / cutlass 4.5.2) up on **B300 (Blackwell Ultra,
sm_103)** inside a real pai-vllm serving process (qwen3.7-max / 3.5-plus, NVFP4
MoE, TP4, paged + varlen). Configuration: head_dim=256, GQA (nqh=32/nkv=2, or
Hq=16/Hkv=1 per TP4 rank), bf16 in/out + fp32 accum, TMEM = 512 columns, SMEM
limit 224 KB.

## 1. Dedicated hd256 2-CTA kernel writes NaN / garbage under concurrent vLLM

### symptom

The dedicated `hd256` 2-CTA kernel (`sm100_hd256_2cta_fmha_forward`) is correct
offline but produces **all-token-0 garbage** end-to-end in vLLM. A same-process
differential probe shows the **kernel itself** writes NaN: `out_nan ≈ 19000`
while the torch reference in the same process has `ref_nan = 0`. The NaNs land on
**49 fixed even output channels**, deterministic per (token, head) and per TP
rank. All inputs are clean (q / attended KV / torch reference all 0 NaN).

The same compiled binary with the same clean input produces **0 NaN in every
offline configuration** tried: JIT-cache on/off, single machine, NCCL 4-process,
dirty SMEM, NaN-poisoning allocator.

### cause

Known FA4 `hd256` TMEM-capacity bug on Blackwell (Dao-AILab issue **#1959**).
The `hd256` accumulator already fills **all 512 TMEM columns**
(`tmem_s_offset=0` + `tmem_o_offset=256`); stacking `P` plus the 2-CTA
cross-CTA accumulation on top overflows capacity → **accumulator corruption →
huge values / NaN**. The 49 even channels are the **peer-CTA half** of the 2SM
MMA (D=256), which is unreliable when other TMEM users (NVFP4 MoE / GDN cutlass
kernels) run concurrently. Offline, FA4 is the *only* TMEM user, so nothing
corrupts it.

### fix

- Route `hd256` through the **general 1-CTA kernel** (`CtaGroup.ONE`,
  `use_2cta_instrs=False`) — no peer-CTA half, no cross-CTA TMEM/DSMEM coherence,
  immune to concurrent corruption (see kernel-opt §1).
- Or mirror upstream vllm-org 0.23's guard: force FA2 fallback when
  `head_size > 128 and != 192` on Blackwell. pai-vllm lacks this guard and
  `VLLM_FLASH_ATTN_VERSION=4` forces FA4, so the corruption reaches production.
- **Do not** try to patch it with a V-SMEM flush: a both-CTA V-SMEM flush drops
  `out_nan` 19000 → 0, but output is still wildly wrong (max|out−ref| ~2e38, 36%
  of finite values > 1e30). The flush hides the NaN symptom, not the
  accumulator-overflow corruption underneath.

## 2. Cheap 1-CTA levers (CLC persistent, split_P_arrive) give null gains

### symptom

On the 1-CTA `hd256` serving path (B=1 prefill, S=7680), the obvious cheap knobs
do nothing:
- CLC persistent scheduling (`FA_CLC=1`, confirmed `scheduling_mode=CLC`):
  0.988 vs 0.991 ms.
- Sweeping `split_P_arrive ∈ {32, 64, 96}` (`FA4_SPLIT_P`): all 0.992–0.998 ms.

Meanwhile ncu @ S7680 vs trtllm-gen: Duration 960 vs 612 µs, Compute SOL 46.8 vs
78.8%, tensor active 44.5 vs 78.1%, occupancy 15.5 vs 18.6% (both low).

### cause

The knobs address the wrong bottleneck. CLC persistent earns its keep by
balancing *many* tiles — a B=1 prefill is already balanced. `split_P_arrive`
tunes *tail* overlap, but the cost is in the *bulk* softmax-wait. The smoking gun
is `stall_barrier` = **7.31 vs 0.002** warps/issue-cycle: with `q_stage=1` (forced
by the 512-column TMEM limit — `2×S=256 + O=256 = 512`; `q_stage=2` would need
768) there is **no second Q-tile to overlap**, so MMA blocks in
`producer_acquire` every KV block waiting for softmax to produce `P`.

### fix

Stop tuning occupancy / launch / tail knobs. trtllm-gen reaches 79% SOL at the
**same 18.6% occupancy**, proving the limiter is pipeline/scheduling efficiency,
not occupancy. The productive levers are structural: `tile_n=96` to unlock
`kv_stage=2` (kernel-opt §2), or adopt trtllm-gen. When occupancy is comparable
but SOL is far apart, the gap is overlap, not occupancy.

## 3. Single-warpgroup S-buffer ping-pong overlap won't compile in 4.5.2

### symptom

Attempting "Option A'" — 2-stage ping-pong of the single softmax-warpgroup's `S`
buffer (write `QK[i+1]` to the other buffer, overlap with `softmax[i]`) to kill
the barrier stall — the **in-process MLIR/LLVM compile explodes** (pure Python at
100% CPU, no `ptxas` subprocess ever spawned). All three coding strategies hang:
- runtime buffer index + if/else copied gemm/pipeline → killed at **29 min**
- fully-runtime single gemm (runtime `acc_tmem_addr` + runtime pipeline
  `w_index`, zero copy) → killed at **7.5 min+**
- `constexpr range_constexpr(2)` unroll + parity if/else → killed at **20 min**

Control: a plain 2-stage pipeline **sizing** (`s_stages=2`) with a normal loop
compiles fast and PASSes (rel_l2 1.86e-3).

### cause

The explosion comes from the **QK-ahead cross-buffer** structure itself —
`commit(nbuf)` while `acquire(buf)` in the same loop — which CuTeDSL 4.5.2 cannot
legalize. It is **not** runtime indexing, **not** branch duplication, and **not**
pipeline sizing: all three were ruled out (the zero-copy fully-runtime version
has default-ish IR size and explodes identically). `q_stage=2` (hd128) compiles
only because its two stages are **two independent Q-tiles** with paired
per-stage acquire/commit; a single warpgroup's KV overlap needs the cross-buffer
pattern the compiler chokes on.

### fix

Do not attempt single-warpgroup cross-buffer ping-pong overlap in CuTeDSL 4.5.2 —
the ~30 min compile cycle makes iteration infeasible, and a minor DSL bump does
not help (4.6.0 hits the same wall on both gemm paths). The only compilable
overlap shape is **paired independent stages** (the `q_stage` form). `tile_n=96`
(kernel-opt §2) is this kernel's reachable ceiling under 4.5.2; for more, use
trtllm-gen or wait for a deeper compiler fix.

## 4. KV-stage pipeline depth has a sharp sweet spot (too shallow is catastrophic)

### symptom

Sweeping the dense 2-CTA kernel's KV multi-buffer depth `kv_stage ∈ {3, 4, 5}`:
- `kv_stage=4` → optimal (57–64% peak).
- `kv_stage=3` → **catastrophic −40%** (drops to ~40% peak).
- `kv_stage=5` → slightly worse (~56%).
- `ex2_emu_res=3` (lower-order exp2 polynomial) → no improvement, slightly worse.

### cause

At `kv_stage=3` latency-hiding is insufficient: TMA loads and MMA cannot overlap
→ pipeline starvation, and the penalty is large. At `kv_stage=5` the extra SMEM
buys **no** occupancy — the kernel is already pinned at 18.75% by the combined
register + SMEM budget — so it is pure waste.

### fix

Find the multi-buffer sweet spot first and treat "deeper is better" as false: too
shallow starves the pipeline (very costly), too deep is a no-op once occupancy is
saturated. `kv_stage=4` for the dense 2-CTA kernel; `kv_stage=2` for the 1-CTA
`tile_n=96` path.

## 5. vLLM NVFP4 / ModelOpt bring-up blockers masquerade as FA4 bugs

### symptom

Before FA4 attention is even reached, loading the NVFP4 MoE model (qwen3.7-max /
3.5-plus) in a non-production pai-vllm checkout (`b46dc08`) crashes twice:
- `VLLM_NVFP4_GEMM_BACKEND=atrex` is an **illegal value** in `b46dc08` (legal:
  `flashinfer-cutlass` / `trtllm` / `cudnn` / `cutlass`) → crash.
- Weight load fails with a quantization conflict: the should-be-BF16
  shared-expert is built as NVFP4 (uint8) and conflicts with the checkpoint.

### cause

- `VLLM_NVFP4_GEMM_BACKEND` is a different switch from `VLLM_NVFP4_USE_ATREX`;
  `setdefault` lets a stale shell-global illegal value leak in.
- `modelopt.py:304-311` only reads the checkpoint's `exclude_modules` → falls back
  to an incomplete `ignore` list (121 entries, **no** shared_expert), and ignores
  the HF-standard `modules_to_not_convert` (664 entries, explicitly listing every
  `shared_expert.{gate,up,down}_proj` as not-to-quantize). Production vLLM
  (`f19c3221`) + atrex handles both; this checkout does not.

### fix

- Explicitly pass a legal GEMM backend (e.g. `flashinfer-cutlass`); do not trust
  a leftover backend env var in the shell.
- Make the modelopt config **merge `modules_to_not_convert` into the exclude
  list** so shared-experts stay BF16.
- Recognize these as version-skew integration traps, not FA4 kernel bugs — clear
  them first, or you will misattribute the crash to attention.

## evidence + reproduction

- Session `4b8ff1d7-...`, branch `fa4_bf16_varlen`, git commit `2b1511f`.
- Trap #1: `FA4_DUMP_PATH` captures the first FA4 call in vLLM, replayed offline
  with a same-process differential NaN probe (`out_nan`≈19000 vs `ref_nan`=0);
  V-SMEM flush experiment shows NaN→0 but max|out−ref| ~2e38. Root cause tracked
  to Dao-AILab issue #1959. Kernel refs `sm100_hd256_2cta_fmha_forward.py:189`,
  `interface.py:978`.
- Trap #2: `FA_CLC=1` / `FA4_SPLIT_P` sweeps + `ncu` deep diff vs
  `trtllm_batch_context_with_kv_cache` (flashinfer 0.6.12).
- Trap #3: three ping-pong implementations, all killed (29 / 7.5+ / 20 min);
  control `s_stages=2` sizing PASS (rel_l2 1.86e-3). Cross-checked on 4.6.0.
- Trap #4: `kv_stage ∈ {3,4,5}` + `ex2_emu_res=3` sweep on 4 production shapes
  (S ∈ {7680, 8064, 8320, 8576}).
- Trap #5: pai-vllm checkout `b46dc08`; `modelopt.py:304-311`.

## affected versions

- CuTeDSL / nvidia-cutlass-dsl **4.5.2** (ping-pong compile explosion also
  reproduced on **4.6.0**).
- FA4 `flash_attn.cute` (`sm100_hd256_2cta_fmha_forward` dedicated kernel;
  general `FlashAttentionForwardSm100`).
- flashinfer 0.6.12 (trtllm-gen baseline).
- pai-vllm `b46dc08` (integration traps), vs production `f19c3221`.
- **sm_103 / B300 (Blackwell Ultra)**, TMEM 512 columns, SMEM 224 KB, bf16 +
  fp32 accum, GQA hd256.
