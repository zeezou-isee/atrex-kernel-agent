# CuTeDSL Paged-Attention Decode on sm_103 (B300) — Optimization Journey

End-to-end journey of a `paged_attention_decode` kernel in **CuTeDSL (cutlass
4.5.2)** on **NVIDIA B300 / sm_103a (Blackwell Ultra)**: from a correctness-first
warp-MMA baseline (696–2399µs) through a cp.async/split-KV warp-MMA anchor (V5,
45.1µs) to a hand-written `tcgen05`/TMEM warp-specialized decode whose 31-shape
mean (44.11µs) slightly beats the best self-authored warp-MMA version and is 3.4×
faster than the first integrated tcgen05 build — closing the gap to the trtllm-gen
baseline to ~1.5× on small shapes.

**Last Updated:** 2026-07-03 · 28 experience records across V0–V10 (warp-MMA path
+ tcgen05 foundation) and the a0–a7 warp-spec productionization.

## Target hardware / DSL

| Item | Value |
|---|---|
| GPU | NVIDIA B300 (Blackwell Ultra) |
| Compute Capability | sm_103 / sm_103a (CC 10.3) |
| SMs | 148 |
| HBM | HBM3e (≈7.2 TB/s achievable, per V4 record) |
| TMEM | 128 lane × 512 fp32 col |
| SMEM max opt-in | 232448 B (`cudaDevAttrMaxSharedMemoryPerBlockOptin`) |
| Tensor path | `tcgen05.mma` / UMMA / TMEM / TMA available (unlike sm_120 client Blackwell) |
| DSL | CuTeDSL cutlass 4.5.2 |

**Workload:** bf16 in/out, fp32 softmax/accumulate; `hd256`; GQA `nqh ∈ {8,16,32}`,
`nkv=2`, `q_per_seq=4`; `block_size ∈ {64,256}`; `block_n=64`; 31 production shapes
(`klen=256` for 15 shapes, `klen=1024` for 16). Varlen causal-tail GQA paged decode.

**Design invariant** (enables single coalesced gather): with `block_n=64` and
`block_size ∈ {64,256}`, every 64-key KV tile lands inside a single physical block
with contiguous `head_dim` → one `(N,D)` 128-bit cp.async gather.

**Baselines** (cudagraph GPU-only, the only trustworthy lens here — see
"Measurement discipline"): vLLM FA varlen 127.6µs; flashinfer trtllm-gen
`fmhaSm100 H256 PagedKv Causal P64 Q16 Kv128 Persistent` recorded at 19.3µs (early
cudagraph mean 19.6µs), later measured at **12.3µs full-path (fmha body 9.3µs)** on
an idle GPU.

## Version ladder

Outcome: ✅ win kept · ➖ neutral/anchor/diagnostic · ❌ dead-end.

| Ver | Outcome | Technique | Measured (B300 sm_103, cudagraph/GPU-only unless noted) |
|---|---|---|---|
| V0 | ➖ | scalar paged gather correctness baseline (warp `MmaF16BF16Op(16,8,16)`, online exp2 softmax) | 696–2399µs across shapes; tflops 3.133, tc_util 0.139%, bw_util 0.70%; 31/31 rel 0.0021 |
| V1 | ✅ | vectorized 128-bit paged gather (single-physical-block invariant) | mean ~196µs (4–10.8× over V0); 543 GB/s big shape |
| V2 | ✅ | drop per-call `cu_seqlens_q.tolist()` sync; enable cudagraph | eager 196→100.7µs (1.95×); cudagraph 67.0µs |
| V3 | ✅ | `data_ptr`-keyed launch cache (skip ~112µs `from_dlpack`×7 rebuild) | eager 66.6µs ≈ cudagraph 67.1µs |
| V4 | ✅ | cp.async 2-stage double-buffered KV pipeline (`CopyG2SOp(GLOBAL,128b)`) | 47.1 / 47.4µs (1.41×); BW 543→1192 GB/s; tflops 71.8; trtllm gap 3.45→2.44× |
| V5 | ➖ | split-KV + combine + cp.async double-buffer (warp-MMA anchor) | eager 45.1 / cudagraph 45.2µs; 31/31 rel 0.00213; 2.8× vs vLLM FA 127.6µs |
| V6 | ❌ | `block_n=32/128` tile-size tuning | FAIL revert: 32→46.2µs; 128→`CUDA_ERROR_INVALID_VALUE` |
| V7 | ➖ | disassemble the 26µs Triton decode PTX | Triton uses `tcgen05.mma`+TMEM; V5 uses warp-MMA; 1.7× gap (research) |
| V8 | ❌ | exhaustive Triton heuristic tuning + prove tcgen05 blocks on SM103 | Triton floor 26.2–26.3µs; `page256` stuck 42–46µs; tcgen05 blocks compile+run |
| V9 | ✅ | tcgen05 QK^T→TMEM→register softmax foundation | single-tile 8.8→11.3µs (launch-bound); 3.2× vs scalar 28.5µs |
| V10 | ✅ | in-kernel P@V fusion cracked (disjoint TMEM S/P/O, M=128) | single-tile 11.2µs (launch-bound); rel_l2 0.0018 |
| stage4 | ➖ | first integrated tcgen05 split-KV decode | mean 149.6µs (max 398.9) |
| a0–a3 | ✅ | warp-specialized decode build (load warp + mbarrier + per-D-chunk TMEM-O rescale) | a3 klen256 rel 0.002102 / klen1024 rel 0.002105 PASS |
| a3 | ✅ | KSTAGES=2 + drop `sQn` (64 KB) | shape0 decode 57.7→30.6µs (1.9×) |
| a4 | ✅ | warp-spec + split-KV (NS=4) + combine | shape0 18.56µs; full mean fp32 47.71µs |
| a4+bf16 | ➖ | bf16 partials (halve combine read BW) | mean 44.11µs (min 15.55 / max 104.84); beats V5 45.3, 3.4× vs stage4 |
| combine | ✅ | standalone combine kernel, `N_DCHUNK` occupancy knob | combine 26.65→10.34µs; full pipeline 53.31→43.15µs |
| a6 | ❌ | FA4-style 10-warp within-CTA pipeline | shape9 decode 100 vs a4 91µs (+10%) |
| a7 | ❌ | N=128 / Kv128 (halve tile count) | shape9 105 vs 104; shape0 22.4 vs 17.3µs |

## Attempts in detail

### V0 — scalar paged gather correctness baseline (➖)

Grid `(B, nkv)`; `qn·rep` packed into the M-tile; element-wise scalar paged gather
of K/V into SMEM; warp `MmaF16BF16Op(16,8,16)` with fp32 accumulate for QK^T and
P@V; online exp2 softmax (mn-view + quad-reduce); causal-tail mask; smem→gmem
store. Latency 696.6µs (shape0) up to 2399.4µs (shape9); tflops 3.133, tc_util
0.139%, bw 50.48 GB/s, bw_util 0.70% (0.08–0.70% SOL); rel 0.002126, 31/31 PASS.
Because latency is nearly flat with batch, the bottleneck is the per-CTA
**uncoalesced scalar SMEM gather**, not HBM bandwidth (`long_scoreboard` dominates).
Note: M must be padded to MMA granularity with padding rows masked; use
`cudaDevAttrMaxSharedMemoryPerBlockOptin` (232448 B) at runtime since gpu-wiki has
no B300 SMEM/SM figure. The recorded attack order: (1) vectorized 128-bit gather,
(2) double/triple-buffer, (3) split-KV for small batch, (4) tcgen05/TMEM to remove
the register-resident `O[M,256]`.

### V1 — vectorized 128-bit paged gather (✅)

Replaced the scalar gather with a single coalesced `(N,D)` block-slice (32 threads
× 8 bf16/row), exploiting the single-physical-block invariant. Mean ~196µs (shape6
222.7µs), 4–10.8× over V0; 543 GB/s big shape; `long_scoreboard` (9.73) collapsed;
rel ~0.002 unchanged. LD count fell ~8×, removing V0's uncoalesced bottleneck — but
exposed a new one: the bottleneck migrated from memory-bound to **latency-bound**
(waves/SM 0.22, ~6.3% occupancy, 235 regs, load not overlapped with MMA).
Rule: vectorized access is the first lever for a memory-bound decode; cash it, then
pivot to latency/occupancy.

### V2 / V3 — kill host sync + cache the prepared launch (✅)

V2 removed `cu_seqlens_q.tolist()` (a per-call device sync that also blocks
cudagraph capture: "Cannot copy between CPU and CUDA during capture"), deriving
`q_per_seq = T//B` sync-free (exact for the uniform contiguous prefix-sums of all
31 shapes); the kernel body is byte-identical to V1. This also unlocked cudagraph.
V3 added a `data_ptr`-keyed `_LAUNCH_CACHE` so the hot path skips the ~112µs rebuild
(`from_dlpack`×7 + `dim_order` + `mark_compact` + `empty_like`) and calls only the
cached executor (~11µs). Result: V2 eager 196→100.7µs (1.95×), cudagraph first
usable 67.0µs (shape6 89.3, shape18 27.2); V3 eager 66.6µs == cudagraph 67.1µs; rel
0.00213. After this, eager wall converges to GPU time and the gap to trtllm is a
pure GPU-kernel problem. Eliminating any per-call GPU→CPU sync is mandatory for
CuTeDSL decode (cudagraph correctness + host savings).

### V4 — cp.async 2-stage double-buffered KV pipeline (✅)

Synchronous 128-bit gather → cp.async 2-stage: gather atom `CopyG2SOp(GLOBAL,128b)`;
`sK`/`sV` become an `N_STAGES=2` staged swizzled layout with the stage as a
runtime-indexable tensor MODE (`sK[None,None,stage]`); prologue loads tile0 +
commit, the loop prefetches tile `n+1`, `cp_async_wait_group(1)` keeps the prefetch
in flight, drains the current group, barriers; MMA/softmax/mask/grid unchanged. Eager
66.6→47.1µs (1.41×), cudagraph 67.1→47.4µs (1.42×); shape6 89.6→59.8; BW 543→1192
GB/s (14.9% SOL), shapes 9/30 → 2030 GB/s (25–28% SOL); tflops 71.8; no shape
regressed; rel 0.00213; trtllm gap 3.45→2.44×. Double-buffering lifted MLP from 1→2
tiles in flight, hiding KV load behind Tensor-Core work — biggest win in the
`page256` 16-tile regime. Constraint: 3-stage (224 KB) exceeds the SMEM opt-in
(232448 B), so cap at 2-stage (160 KB). Residual: small grid (waves/SM 0.22) +
register/TMEM ceiling (254 regs, register-resident `O[M,256]`, no tcgen05) → V5's
split-KV and the eventual tcgen05 rewrite.

### V5 — split-KV + combine + cp.async double-buffer, warp-MMA anchor (➖)

split-KV + combine reduction + cp.async 2-stage double-buffer; both GEMMs on
warp-MMA (mma.sync register path, not tcgen05/TMEM), fp32 register-resident
accumulator; KV via 128-bit cp.async pack-GQA paged gather. eager 45.1µs / cudagraph
45.2µs, 31/31 correct, max rel_l2 0.00213 — exactly reproducing the reference
45.1/45.3µs. ~2.8× faster than vLLM FA varlen (127.6µs); still short of trtllm-gen
19.3µs. This is a memory-bound form where warp-MMA does not saturate the Tensor
Cores — the starting point for the tcgen05 in-kernel fusion. When reproducing a
baseline, align rel_l2 and latency digit-for-digit to confirm the environment.

### V6 — block_n tile-size tuning (❌)

On V5, tried `block_n=32` and `block_n=128` to lift occupancy off the 254-reg /
6.2%-occupancy wall. FAIL, reverted: `block_n=32` regressed to 46.2µs (tile count
doubles, per-tile overhead beats the occupancy gain; big shapes 60→75µs);
`block_n=128` threw `CUDA_ERROR_INVALID_VALUE` (2×128×256×2B/stage over SMEM
opt-in). Root cause: on large shapes `grid=64 < 148 SMs`, so occupancy is
**GRID-limited**, not SMEM/reg-limited — no tile knob can place a 2nd block per SM.
Transferable rule: when `grid_CTAs < num_SMs`, occupancy is fixed by the grid; either
split-KV to add CTAs or change the MMA path structurally (tcgen05/UMMA+TMEM to move
`O[M,256]` out of registers). V5 is the warp-MMA architecture ceiling; the residual
2.33× to trtllm needs a tcgen05 rewrite.

### V7 — disassemble the Triton PTX (➖)

Disassembled the 26µs Triton decode kernel's SM103 PTX to explain its 1.7×
advantage over V5. Finding: Triton already emits `tcgen05.mma` + TMEM
(`tcgen05.alloc` 512 cols, QK^T 4× `tcgen05.mma`, P@V 16×,
`tcgen05.ld/st.16x32bx2` for TMEM↔reg softmax readback / P-O writeback, cp.async
KV, mbarrier pipeline), while V5 uses legacy `warp.MmaF16BF16Op` (mma.sync +
register accumulate). The gap is a hardware-path difference, not tunable —
register accumulation pins `O[M,256]` in registers (254 regs), a structural
ceiling. Fix direction: switch CuTeDSL explicitly to `tcgen05.MmaF16BF16Op` +
`TmemAllocator` + `make_smem_layout_a/b` to reproduce Triton's auto-hit path.
Method: when an auto-compiler is much faster on the same hardware, read its PTX
first to see which instruction path it uses.

### V8 — Triton exhaustive tuning + prove tcgen05 blocks (❌ + foundation)

(a) Exhaustive Triton heuristic tuning (split sweep sp=1..4, `num_warps`,
`packed_base`) bottomed at 26.2–26.3µs (shapes 16,17 improved 3.8–5.9µs), but the
large rep=16 `page256` shapes (7–9, 28–30) stayed stuck at 42–46µs, 254 regs, ~6%
occupancy, instruction-throughput-limited — `BLOCK_N`/stages/warps/splits move
nothing. (b) Proved the CuTeDSL tcgen05 building blocks compile and run on SM103:
scalar SMEM write + `tcgen05.MmaF16BF16Op(64,64,16)` + TMEM accumulator +
`Ld16x64bOp` readback. Rule: an auto-compiler's tuning space has a hard upper bound;
past it, hand-written tcgen05 is required — and the blocks are now proven, paving
V9/V10.

### V9 — tcgen05 QK^T → TMEM → register softmax foundation (✅)

Single-tile M=64 N=64 D=256: vectorized 128-bit cp.async into swizzled tcgen05 SMEM
→ `tcgen05.MmaF16BF16Op(64,64,16)` QK^T → TMEM → `Ld16x64bOp` readback → `*scale_log2`
→ `exp2`, single CTA. Vectorized cp.async+MMA 8.8µs (vs scalar 28.5µs, 3.2×,
`de317cd`); K=256 full QK^T 11.2µs (`e263755`); QK+scale step1 10.9µs (`37b3e95`);
+exp2 step2 11.3µs PASS (`ac54f15`). Proves tcgen05 QK^T + TMEM readback +
register-side softmax numerator (scale, exp2) work on SM103. Still missing: softmax
row reduction, P@V, paged gather, multi-tile online-softmax loop. Metering note:
11.3µs is a single-CTA tiny-tile launch-bound value — not comparable to trtllm
(19.6µs, 31 shapes, multi-CTA). This is the last reusable foundation before in-kernel
fusion (V10).

### V10 — in-kernel P@V fusion cracked (✅)

Cracked the in-kernel tcgen05 PV fusion that prior V7–V10 attempts never ran.
Single-tile M=128 N=64 D=64 isolation, base `9a14c6d`. Four rules:
1. **S/P/O in DISJOINT TMEM column regions** (`S[0,64)`, `P bf16 [64,96)`,
   `O[128,384)`); O zeroed once before the loop, then accumulated. `ACCUMULATE=False`
   only resets the first k-block of the *next* MMA, not a cross-tile accumulator —
   conflating this was the historical `rel_l2=1.5`.
2. **M-tile aligned to 128** via `make_trivial_tiled_mma` for a clean fragment
   (M=64 gives a composite `(16,4)` that `Ld32x32`/`St32x32` reject).
3. **P bridged back to TMEM**: `Ld32x32bOp` read S, register `exp2`, pack bf16 P into
   an fp32 word (`rP_bf16` sharing the `rP_f32` register) via `St32x32bOp(Float32)`;
   P as PV A-operand uses the **fp32 S iterator + bf16 A-layout** (never
   `recast_ptr(bf16)`, which doubles the M-stride 131072 vs 65536).
4. PV MMA runs TS-mode, `a_major=K`.

Full single-tile FMHA (QK→exp2 softmax→PV→O readback) `rel_l2=0.0018` PASS (v1
unnorm; v2 with real softmax still 0.0018). `do_bench` 11.2µs but LAUNCH-BOUND
(single CTA tiny tile ≈ QK-only 11.4µs, GPU compute <1µs) — a **correctness
breakthrough**, not a perf figure.

### Foundation records (fold into V9/V10 build-up)

- **Thread-local softmax reduction (N=64 sweet spot, ✅):** with `Ld32x32bOp` reading
  a `(128,64)` S tile, each of 128 threads holds one full N=64 row (128×64 = M·N), so
  online-softmax row max/sum and per-thread O correction are entirely thread-local —
  no cross-lane shuffle, far simpler than FA4's warp-specialized `SoftmaxSm100`.
  Multi-tile online softmax 2-tile and 4-tile both `rel_l2 ≈ 0.001` PASS. Register
  correction `O = O*corr + O_t` works at D=64; at D=256 it must move to TMEM (below).
- **PV V-orientation rule (✅):** a deterministic probe (`V[n,d]=d+1`, P selecting key
  0) proved tcgen05 PV MMA computes `P @ (sV)^T` **regardless of `b_major`** — natural
  V gives `O=[1,1,1,…]=P@V^T` (wrong), V^T gives `O=[1,2,3,…]=P@V` (right, rel 0.0018).
  So `sV` must physically store V^T; `autovec_copy` is an identity copy, and cp.async
  cannot transpose (need `sVn_T` stride-swapped view / `ldmatrix.trans` / TMA). Resolve
  orientation puzzles with a value probe, not blind `b_major` sweeps.
- **hd256 TMEM-O budget (➖ structural constraint):** at D=256, O has 256 values/row →
  can't fit registers (>254) and its TMEM accumulator alone owns **256 of 512 columns**;
  `S(64) + O(256) = 320 ≤ 512` → single-buffer only. Single-tile can normalize P
  before PV (P is 64/row) and write O directly (`rel_l2=0.0018`, 21.5µs); multi-tile at
  D=256 **must** rescale O per D-chunk in TMEM (`Ld16x64`/`St16x64`). This budget
  drives all tile-size/buffering decisions.

## Productionization — tcgen05 warp-specialized decode (a0–a7)

The V10 correctness breakthrough was integrated toward production in a second
session, mirroring the stage1→4 progression.

### stage4 — first integrated tcgen05 split-KV decode (➖)

The from-scratch integration reached full-31-shape correctness but was slow: mean
149.6µs (max 398.9µs). Note trap #1 (ref pitfalls): the all-klen single-CTA variant
failed LLVM module serialization — a 4-tile constexpr unroll of a heavy tcgen05 body
overflowed the module; `num_tiles=1` bisected it. Split-KV is therefore the
**compile-time prerequisite**, not just perf.

### a0–a3 — warp-specialized decode build (✅)

Built step-by-step: a0 = dedicated LOAD warp (cp.async producer) + MMA warp
(tcgen05 consumer) + hand-written mbarrier 2-stage double-buffer QK skeleton (last-tile
`rel=0.0`, clearly distinct from tiles 0–2, proving staging delivers correct K); a1 =
full QK→online-softmax→PV→O mainloop interleaved into the COMPUTE group (warps 0–3),
`load(t+1)` overlapping `compute(t)`, 4-tile `rel=0.0014`; a2 = real D=256 with
per-D-chunk `Ld32x32`/`St32x32` TMEM-O rescale, `rel=0.0011`; a3 = real paged input
(GQA Q-pack + paged gather + V transpose + causal-tail), klen256 `rel=0.002102`,
klen1024 (16 tile) `rel=0.002105`, all constexpr loops (~few-minute compile, no IR
blowup thanks to the lean per-tile body). Distilled rules: (1) the load-warp cp.async
must be tiled by its **real 32 threads** (`lane=tidx%32`), not 160 (symptom
`nonzero_rows=112`); (2) tcgen05 tmem-copy atoms **cannot cross an `scf` region**
(dynamic `for` or warp-spec `if`), can't be re-partitioned to another offset, can't
capture closures → create them in-region per-offset; (3) intra-compute-group sync uses
`NamedBarrier(128)`, not `__syncthreads` (else deadlock vs the load warp); (4)
per-D-chunk `Ld32x32` puts corr naturally per-row in each thread, avoiding an `sScale`
broadcast; special-case the first tile `ACCUMULATE=False` to avoid NaN×0; (5) aliased
SMEM (`sVn` over `sQn`) needs a barrier under warp-spec — the load warp's tile-0
V-gather overwrote Q the compute warp was still reading (symptom: uniform ~0.245 error
across all queries/heads, the overwrite signature); fix = move Q-pack before the
load/compute split + `sync_threads()`.

### a3 KSTAGES=2 + drop sQn (✅)

Dropped the 64 KB `sQn` staging (Q gathered per-thread from gmem into registers, then
relayout into swizzled sQ), freeing SMEM to open KSTAGES=2 double-buffer so `load(t+1)`
overlaps `compute(t)`. shape0 decode-path 57.7→30.6µs (1.9×, first to beat stage4's
43µs), rel 0.002102. Trap #10: the launch SMEM request must equal the SM100 max opt-in
(232448; actual ~229376) — a request below actual (228000) does not fail compile, it
faults at runtime (silent OOB). Push SMEM to `cudaDevAttrMaxSharedMemoryPerBlockOptin`.

### a4 split-KV — occupancy beats within-CTA overlap (✅)

Placed the a3 warp-spec decode into the stage4 split-KV skeleton (`grid=(B,nkv,num_splits)`,
writes partials, reuses combine). shape0: NS=4 (1 tile/split, 64 CTA) = 18.56µs <
NS=2 (2 tile, 32 CTA, cross-tile overlap) = 22.5µs < NS=1 (4 tile, 16 CTA) = 32µs.
Decompose: decode 15.3 + combine 10.5 = full ~19µs. Trajectory: stage4 43 → a3 30.6 →
a4 18.56µs, approaching trtllm 12.3µs (1.5×). For small klen, raising occupancy
(split-KV to ~64 CTA) **decisively beats** cross-tile overlap — KSTAGES=2 needs ≥2
tiles/CTA, so the two are mutually exclusive. This is exactly trtllm's recipe:
occupancy via split-KV (1 tile/split), single-tile latency hidden by more warps.

### combine kernel — N_DCHUNK occupancy knob (✅)

Wrote the flash-decoding combine as its own kernel, `grid=(B,nkv,N_DCHUNK)`: phase-1
global-max + per-split rescale, phase-2 D-chunk weighted accumulate + bf16 O (GQA
scatter). Combine 26.65µs (N_DCHUNK=4, 64 CTA) → 18.30 (8) → 12.43 (16) → 10.34µs
(32, 512 CTA), plateauing at ≥32; full pipeline 53.31→43.15µs, 31/31 PASS rel≈0.0021.
Small-work reductions are occupancy-bound — split D into more CTAs until the SMs fill.
Trap #11: `N_DCHUNK` must divide D=256 exactly (`48→DC=5` covers only 240 cols, silent
bug); use 32 (DC=8).

### Full-shape mean + bf16 partials (➖ correction + hypothesis-kill)

Ran the full 31 shapes (the earlier 18.5µs was only a klen=256 small-B special case).
a4 fp32 mean 47.71µs (min 16.4 / max 116.3); switching partials to bf16 (halving
combine read BW) → 44.11µs (min 15.55 / max 104.84), rel 0.0021→0.0029 (negligible).
Four-way: a4 44.11 / V5 45.3 (max 61.5) / stage4 149.6 (max 398.9) / trtllm ~19.3µs.
So the true mean already slightly beats the best self-authored V5 and is 3.4× faster
than stage4. bf16 **disproved the "combine bandwidth wall"**: it only cut klen1024/B56
by ~11% (116→104), and the decode/combine split of shape9 = decode 90.97 + combine
19.89µs (decode 87%). Lesson: report full-shape mean/max (single-shape peaks mislead),
and profile the decode/combine split before trusting an intuited wall — a cheap bf16
experiment falsifies or confirms it.

### a6 / a7 — within-CTA overlap and bigger tiles (❌)

Two full new kernels chasing mean <30µs: a6 (FA4-style 10-warp pipeline: dedicated
mma/softmax/correction/load warps, S/P TMEM double-buffer, corr via `sScale` smem
broadcast) — shape9 decode 100 vs a4 91µs (+10%); a7 (N=128/Kv128, halving tile count)
— shape9 105 vs 104, shape0 22.4 vs 17.3µs. Both correct, neither faster. a6's root
cause: the **O accumulator is a serial dependency chain** (each tile's PV needs the
prior tile's O rescaled) — more warps/stages can't shorten it, only add barrier cost.
a7's: the kernel is latency-bound, so reducing work doesn't reduce unhidden latency,
and N=128 brings a bigger MMA + 2-half PV + 2-block gather that cancel the tile halving.
Two paired traps: (a6) an `elect_one` inside `if warp_idx<SOFTMAX_WARPS` makes 4 warps
each arrive once on a count-1 barrier → parity flips → mma's wait never satisfies →
deadlock (arrive exactly once); (a7) `block_size=64` with N=128 spans two physical
pages — gather each 64-key sub-block from its own `block_table` entry or keys 64–127
read adjacent garbage.

## Diagnostics that redirected the work (neutral)

### ncu: trtllm 16-warp vs our 4-warp — the decisive redirect

Profiling trtllm-gen vs our tcgen05 decode on shape0 (klen256, B=8, nqh=32): trtllm
full-path cudagraph 12.3µs (fmha body 9.3µs) on idle GPU7 (the old 19.6µs was measured
on a busy GPU5). Both use **64 CTA, waves/SM=0.43, no cluster, no 2-CTA MMA** — the
only difference is 512 threads / **16 warp** (trtllm) vs 128 threads / **4 warp**
(ours). Our decode 39µs vs trtllm fmha 9.3µs. At the same occupancy, 16 warps hide the
MMA/TMEM async latency; 4 warps (warp0 doing QK→softmax→PV serially, zero overlap)
cannot — this is the ~4× gap. It overturned the "raise occupancy / add cluster / 2-CTA
MMA" plan in favor of **more warps per CTA + warp-specialize overlap**. When occupancy,
CTA count and cluster all match but latency differs by multiples, the culprit is
per-CTA warp count (latency-hiding capacity), not occupancy.

### Latency-bound TMEM occupancy wall

ncu on the real work point shape9 (NS=8, ~89.8µs): waves/SM=3.03 (occupancy fine),
`sm_throughput 7.28%`, `warps_active 7.78%` (vs trtllm 21%, ~3× headroom),
`long_scoreboard` = 61% of stall cycles. Levers: within-CTA pipeline −10%; `tps=1`
−20% (offset by doubled fixed cost); vectorize Q/O ≈0%; skip Q-gather −3.4µs, skip
O-write −8.6µs. Root cause: compute-warp TMEM traffic (S read + O rescale read/write +
O readback) exposed at 5 warp / 1 CTA per SM. All three latency-hiding routes are hard
resource-blocked: (1) 2 CTA/SM impossible — not only SMEM (`sQ64+sK+sV=128KB>116KB`)
but **TMEM** (one CTA's O owns 256 of 512 cols; two CTAs' O won't fit — even M=64 can't
save it); (2) batching 4 O D-chunks needs 256 reg/thread (now 108) → register blowup;
(3) vectorization cuts instructions, not latency. Under latency-bound + 1 CTA/SM, only
more concurrent warps help; check both SMEM and TMEM for the occupancy ceiling.

### TMA tensor-descriptor stall (dead-end, but reframed the bottleneck)

~30 from-scratch tensor-TMA paged-KV load PoC variants all device-hang (even
fire-and-forget → the copy instruction itself faults). Isolation: `expect_tx` without a
copy completes; the FA4 forward runs in-env (CC 10.3, 0.045ms); `cp.async.bulk`
(non-tensor, no descriptor) succeeds standalone (`max_err=0`). The fault is in the
tensor-descriptor path (`make_tiled_tma_atom_B`), not async-bulk/mbarrier/env; it only
works inside FA4's complete persistent/warp-spec context. Important reframe: since
`cp.async.bulk` works, **load was never the real bottleneck** — a plain bulk-copy of
re-paged-128 KV pages suffices; the real bottleneck is the compute-warp structure.

### FA4 hd256 reuse — prefill-only (dead-end)

FA4's hd256 2-CTA `flash_attn_varlen_func` (+ `page_table`, re-paged to `page_size=128`)
is correct (`rel=0.0021`) but ~4× slower (klen1024 ~400µs vs a4 104µs); `num_splits>1`
errors (no SplitKV); `pack_gqa` no difference (398 vs 404µs). It's a prefill kernel —
no SplitKV / decode optimization — so on `q_len=4` decode it is grid-underutilized.
There is no off-the-shelf D=256 decode-optimized kernel to reuse; a <30µs decode must
be built as a trtllm-level kernel from scratch.

### The dynamic-loop TMEM-copy limit (dead-end)

Running 16 tiles per CTA with a dynamic `cutlass.range(unroll=1)` + loop-carried
online-softmax state + in-loop `make_tmem_copy` fails MLIR/ICE
(`atom.tmem_load→tiled_copy remained live` in the dynamic `scf.for`), for every variant
tried. cutlass 4.5.2 cannot legalize a tmem-copy atom across a dynamic `scf.for`
boundary (same root as the warp-spec `scf.if` limit). Workaround: `range_constexpr`
specialized by `num_tiles` (production has only 4/16) — verified D=64 16-tile rel=0.0013
and D=256 16-tile PASS, no IR blowup, since the lean warp-spec per-tile body keeps the
constexpr unroll small.

### V10 readback dst-shape bug fix (correctness win)

The Stage4 split-KV "PV→O=0" bug (rel_l2=1.000000) was fixed to 0.002025 (then all
31/31 PASS). Real cause: the O readback register tensor was created from the **source
(TMEM) partition shape** instead of the **destination (gmem) shape**, so the `Ld16x64`
epi-copy silently read back 0 — the only difference from the passing stage3a. All prior
probes were overwritten by the readback (which fills all of `mOp[0,0,0]`), looking like
"binding broken / PV=0". Breaking method: an independent `Ld32x32` per-row read of
TMEM_O returned `sumsq=32.14`, proving PV was always correct and localizing the bug to
readback. Two latent bugs also fixed: masked padding rows → `tile_max=-inf` →
`exp2(-inf-(-inf))=NaN`, clamped with `fmax(tile_max,-1e30)`; and a debug-deleted V
transpose (natural V → P@V^T, rel=1.39) restored. Lesson: build tcgen05 readback
fragments from the **destination** shape, and confirm upstream truth via an independent
read path before trusting a self-overwriting buffer.

## Measurement discipline

On small decode shapes the eager wall-clock is dominated by host launch, not the
kernel: `from_dlpack`×7 + `dim_order` + `mark_compact` ~112µs/call + ~60µs compiled
dispatch, plus ~50–60µs per `.item()`/`.tolist()` sync. This is not an irreducible
framework floor — caching the prepared launch by `data_ptr` makes eager wall ==
cudagraph == GPU-only (the earlier "eager can't beat trtllm's lean C++ launch"
conclusion was wrong). trtllm eager is a constant ~55–68µs (masked by
`seqused_k.max().item()` host sync, shape-independent), while its cudagraph GPU-only is
19.6µs mean (scaling with KV, matching the recorded 19.3µs), and V5 eager 45.1 ≈
cudagraph 45.2. Rules: (1) judge CuTeDSL decode with cudagraph/GPU-only, not eager
`do_bench`; (2) eliminate every per-call GPU→CPU sync (cudagraph correctness + ~hundreds
of µs host savings); (3) reproduce a baseline under the same GPU-only lens — trtllm's
19.3µs only reproduces GPU-only, and looks like ~60µs under eager. This matches vLLM's
production practice of cudagraph-capturing decode.

## Sustained recipe (for any B300 sm_103 CuTeDSL paged decode)

1. Establish a correct warp-MMA baseline, then **vectorize the paged gather** (128-bit
   single-physical-block slice) — the first lever for memory-bound decode.
2. **Kill all per-call GPU→CPU sync** and cache the prepared launch → cudagraph +
   eager==GPU-only; judge with cudagraph/GPU-only only.
3. Add a **cp.async 2-stage double-buffer** KV pipeline (cap stages at the SMEM opt-in).
4. When `grid_CTAs < 148 SMs`, don't tune tiles — **split-KV** to widen the grid, and
   move the flash-decoding **combine** into its own kernel with an `N_DCHUNK` occupancy
   knob (divisor of D).
5. To break the warp-MMA ceiling, rewrite onto **tcgen05/UMMA + TMEM** (read a fast
   auto-compiler's PTX to confirm the target path). Choose **N=64** so softmax is
   thread-local.
6. For in-kernel P@V fusion: **disjoint TMEM S/P/O**, **M=128 clean fragment**, **P
   A-operand via fp32 iterator + bf16 layout**, and **store V transposed** (PV computes
   P@(sV)^T). Build readback fragments from the **destination** shape.
7. At D=256, respect the **TMEM-O budget** (O owns 256 of 512 cols → single-buffer,
   per-D-chunk TMEM rescale, and a 2-CTA/SM wall harder than SMEM).
8. Under latency-bound + 1 CTA/SM, the only lever is **more concurrent warps per CTA**
   (16-warp warp-spec) — work-reduction, vectorization, and within-CTA overlap of a
   serial O-accumulator are all no-ops.

## Related docs

- **Quick reference (proven techniques):**
  [`docs/kernel-opt/nvidia/cutedsl/sm103/sm103-paged-attention-decode.md`](../../../../kernel-opt/nvidia/cutedsl/sm103/sm103-paged-attention-decode.md)
- **Pitfalls (14 traps → symptom → cause → fix):**
  [`docs/pitfalls/nvidia/cutedsl/sm103-paged-attention-decode-pitfalls.md`](../../../../pitfalls/nvidia/cutedsl/sm103-paged-attention-decode-pitfalls.md)
