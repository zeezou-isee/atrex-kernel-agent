# SM103 Paged-Attention Decode: tcgen05 + split-KV + cp.async Quick Reference (kernel-opt)

> This is an **optimization highlights quick reference**. See "Further Reading" for the full v0→v10 + warp-spec productionization journey and the pitfalls collection.

Target: `paged_attention_decode` in **CuTeDSL (cutlass 4.5.2)** on **NVIDIA B300 / sm_103 (Blackwell Ultra, HBM3e, 148 SMs)**.
Workload: bf16 in/out, fp32 softmax/accumulate, `hd256`, GQA `nqh ∈ {8,16,32}`, `nkv=2`, `q_per_seq=4`, `block_size ∈ {64,256}`, `block_n=64`, 31 production shapes (`klen` 256 for 15 shapes, 1024 for 16). B300 exposes `tcgen05.mma` / UMMA / TMEM (128 lane × 512 fp32 col) / TMA — unlike sm_120 client Blackwell.

## Trigger Conditions

This decode kernel is small per-CTA and moves through three distinct bottleneck regimes as you optimize. Use this reference when ncu / timing shows any of:

- **Latency stays flat with batch** (e.g. 696µs → 2399µs is only a 3.4× spread while batch grows far more) → per-CTA **uncoalesced scalar paged gather**, not HBM bandwidth. Go to §1.
- **eager `do_bench` wall ≫ GPU time** (host launch ~112µs `from_dlpack`×7 + `.item()`/`.tolist()` sync) → launch-bound, not kernel-bound. Go to §2.
- **`long_scoreboard` dominates warp stalls** with a multi-tile `page256` shape → KV load latency exposed behind MMA. Go to §3.
- **`grid_CTAs (B·nkv) < 148 SMs`, ~6% occupancy, waves/SM ≈ 0.22–0.43** → grid-limited occupancy. Go to §4.
- **A Triton build is ~1.7× faster on the same shape** → it is auto-emitting `tcgen05.mma` + TMEM while your kernel is on legacy `warp.MmaF16BF16Op` (mma.sync + register-resident `O[M,256]`). Go to §6–§9.
- **`waves/SM ≈ 3` yet `sm_throughput < 8%` and `warps_active ≈ 8%` (vs a fast kernel's ~21%)** → latency-bound with too few warps per CTA to hide TMEM read latency. This is the final wall; see Anti-Patterns.

## The Optimization Set (Apply in Order)

### 1. Vectorized 128-bit paged gather (memory-bound → latency-bound)

Replace element-wise scalar SMEM gather with a **single coalesced 128-bit block-slice** load (32 threads × 8 bf16 per row). This depends on one design invariant: with `block_n=64` and `block_size ∈ {64,256}`, every 64-key KV tile falls inside a **single physical block** with contiguous `head_dim`, so one `(N,D)` block-slice is legal. This is the first lever for a memory-bound decode; it cut LD count ~8× and collapsed `long_scoreboard`. Cash the win, then immediately pivot to latency/occupancy (it exposes small-grid / ~6% occupancy).

### 2. Kill per-call GPU→CPU sync + cache the prepared launch

Two host-side fixes that also unlock CUDA-graph capture and make eager wall converge to GPU time:
- Remove `cu_seqlens_q.tolist()` / any `.item()` — each forces a device sync (~50–60µs) **and** aborts cudagraph capture ("Cannot copy between CPU and CUDA during capture"). Derive `q_per_seq = T//B` sync-free (exact for the uniform contiguous prefix-sums of all 31 shapes).
- Cache the prepared launch keyed by `data_ptr` so the hot path skips the ~112µs rebuild (`from_dlpack`×7 + `dim_order` + `mark_compact` + `empty_like`) and calls only the cached compiled executor (~11µs).

After this, eager == cudagraph == GPU-only, and the gap to trtllm becomes a pure GPU-kernel problem.

### 3. cp.async multi-stage KV pipeline

Convert the synchronous 128-bit gather to a **2-stage double-buffered cp.async** software pipeline: gather atom = `CopyG2SOp(cache_mode=GLOBAL, num_bits=128)`; `sK`/`sV` become an `N_STAGES=2` staged swizzled layout with the stage encoded as a **runtime-indexable tensor MODE** (`sK[None,None,stage]`, not a per-stage Python list); prologue loads tile0 + `cp_async_commit_group`, the loop prefetches `n+1`, `cp_async_wait_group(1)` keeps the prefetch in flight, drains the current group, barriers. This lifts memory-level parallelism from 1→2 tiles in flight and hides KV load behind Tensor-Core work; the win is largest in the `page256` 16-tile regime. Constraint: 3-stage (224 KB) exceeds the SMEM opt-in (232448 B), so cap at 2-stage (160 KB).

### 4. Split-KV for occupancy — it beats within-CTA overlap for small klen

When `grid_CTAs < 148 SMs`, occupancy is **grid-limited**: no tile/stage/register tuning can place a 2nd block per SM. The fix is structural — split-KV widens the grid from `B·nkv` to `B·nkv·num_splits` and writes partials for a combine pass. On small `klen`, raising occupancy (NS=4 → ~64 CTAs, 1 tile/split) **decisively beats** cross-tile overlap (KSTAGES=2 needs ≥2 tiles/CTA — the two are mutually exclusive). Measured on shape0: NS=4 = 18.56µs < NS=2 = 22.5µs < NS=1 = 32µs. This is exactly trtllm's recipe: occupancy via split-KV, single-tile latency hidden by more warps. Split-KV is also a **compile-time prerequisite** — see §5 and the pitfalls doc for why one CTA cannot constexpr-unroll all 4–16 tiles.

### 5. Standalone combine kernel with an N_DCHUNK occupancy knob

Write the flash-decoding reduction as its own kernel, `grid=(B, nkv, N_DCHUNK)`: phase-1 computes global-max + per-split rescale, phase-2 does the D-chunk weighted accumulate. This reduction is small-work and occupancy-bound, so sweep `N_DCHUNK` to fill the SMs: combine dropped 26.65µs (N_DCHUNK=4, 64 CTA) → 10.34µs (N_DCHUNK=32, 512 CTA), plateauing at ≥32. Hard rule: **N_DCHUNK must divide D=256 exactly** — `48 → DC=5` covers only 240 columns (a silent correctness bug); use 32 (DC=8).

### 6. tcgen05 QK^T → TMEM → register-side softmax (foundation)

The warp-MMA path is a hard architectural ceiling (see Anti-Patterns / V6). The structural fix is to move onto the same path Triton auto-hits: `tcgen05.MmaF16BF16Op(64,64,16)` for QK^T → result in TMEM → `Ld16x64bOp`/`Ld32x32bOp` readback to registers → `*scale_log2` → `exp2`. Proven correct on SM103 and 3.2× faster than the scalar variant (8.8µs vs 28.5µs single-tile). Build blocks: vectorized 128-bit cp.async into swizzled tcgen05 SMEM, `TmemAllocator`, `make_smem_layout_a/b`.

### 7. Thread-local softmax reduction at N=64 (the sweet spot)

With an `Ld32x32bOp` readback of a `(128,64)` S tile, all 128 threads each hold **one full row** of N=64 values (128×64 = M·N). So online-softmax row max/sum and per-thread O correction are **entirely thread-local** — no cross-lane shuffle, far simpler than FA4's warp-specialized `SoftmaxSm100`. Pick N=64. The register-resident correction `O = O*corr + O_t` is valid at D=64; at D=256 O overflows registers and must rescale in TMEM (§ hd256 budget, below).

### 8. In-kernel P@V fusion (the V10 breakthrough)

Fusing PV in-kernel (never landed in prior V7–V10 attempts) rests on three rules:
- **S, P and O occupy DISJOINT TMEM column regions** (e.g. `S[0,64)`, `P bf16 [64,96)`, `O[128,384)`). O is zeroed once before the loop and accumulated. `ACCUMULATE=False` only resets the first k-block of the *next* MMA — it does **not** clear a cross-tile accumulator; conflating this caused the historical `rel_l2=1.5`.
- **M-tile aligned to 128** via `make_trivial_tiled_mma` for a clean fragment. `M<128` yields a composite `(16,4)` fragment that the `Ld32x32`/`St32x32` atoms reject.
- **P as PV A-operand: use the fp32 S iterator + a bf16 A-layout**, never `recast_ptr(bf16)` on the TMEM pointer (that doubles the M-stride 131072 vs 65536). Bridge P back through TMEM by packing bf16 into an fp32 word (`rP_bf16` sharing the `rP_f32` register) via `St32x32bOp(Float32)`. PV MMA runs TS-mode, `a_major=K`.

Single-tile fused FMHA (QK→exp2 softmax→PV→O readback) reaches `rel_l2=0.0018`. (The 11.2µs number is launch-bound, single-CTA tiny tile — a correctness result, not a perf figure.)

### 9. PV V-orientation rule — store V transposed

tcgen05 PV MMA computes **P @ (sV)^T regardless of `b_major`** (both K and MN give P@V^T). To get attention's P@V, `sV` must **physically store V^T**. `autovec_copy` with the same TV layout copies in physical order (identity, no logical transpose); a real transpose needs a stride-swapped source view (`sVn_T`), `ldmatrix.trans`, or TMA — cp.async cannot transpose. Feeding natural V yields `O=[1,1,1,…]=P@V^T` (wrong); feeding V^T yields `O=[1,2,3,…]=P@V` (right, rel 0.0018). Resolve orientation puzzles with a deterministic probe (`V[n,d]=d+1`, P selecting key 0), never blind `b_major` sweeps.

### 10. hd256 TMEM-O budget (structural constraint)

At D=256, O has 256 values/row → cannot fit registers (>254) and its TMEM accumulator alone owns **256 of the 512 columns**. So `S(64) + O(256) = 320 ≤ 512` → **single-buffer only**, and the register-resident O trick is invalid. Single-tile can normalize P before PV (P is only 64/row) and write O directly; multi-tile online-softmax at D=256 **must** rescale O per D-chunk in TMEM (`Ld16x64`/`St16x64` read → *corr → write). This budget drives all tile-size / buffering decisions and is the harder-than-SMEM wall against 2 CTAs/SM.

### 11. Warp-specialized decode build + KSTAGES=2

Assemble the production kernel as: a **dedicated LOAD warp** (cp.async producer) + a COMPUTE group (warps 0–3 interleaving QK→online-softmax→PV→O) + hand-written mbarrier 2-stage double-buffer; per-D-chunk `Ld32x32`/`St32x32` for TMEM-O rescale. Then KSTAGES=2 + dropping the 64 KB `sQn` staging (Q gathered per-thread from gmem into registers, relayout into swizzled sQ) overlaps `load(t+1)` with `compute(t)`: shape0 decode 57.7µs → 30.6µs (1.9×, first to beat the stage4 integrated kernel). Required patterns:
- Load-warp cp.async must be tiled by its **real 32 threads** (`lane = tidx%32`), not the full 160 (else K/Q under-loaded — symptom `nonzero_rows=112`).
- Every tcgen05 tmem-copy atom must be created **in-region, per-offset** — it cannot cross an `scf` boundary (dynamic `for` or warp-spec `if`), be built from one tensor and re-partitioned to another offset, or capture a closure.
- Intra-compute-group sync uses `NamedBarrier(128)`, **not** `__syncthreads` (which deadlocks against the load warp).
- Set the launch SMEM request to the SM100 max opt-in (`232448`; actual need ~229376) — a request slightly below actual does **not** fail to compile, it faults at runtime (silent OOB).

### 12. bf16 partials — a free win, and a hypothesis-killer

Storing split-KV partials as bf16 (halving combine read BW) drops the full-31-shape mean from 47.71µs → 44.11µs (rel 0.0021 → 0.0029, negligible) and partial memory halves. It also **disproves the "combine bandwidth wall"**: bf16 only helped klen=1024/B56 by ~11% (116→104µs), and the decode/combine split shows shape9 = decode 90.97µs + combine 19.89µs (decode is 87%). Always profile the split before trusting an intuited wall.

## Measured Benefits

### Warp-MMA path (cutlass 4.5.2, 31 shapes; eager / cudagraph)

| Version | Technique | Latency | Note |
|---|---|---|---|
| V0 | scalar paged gather correctness baseline | 696–2399 µs across shapes | 31/31, rel 0.0021; tc_util 0.139%, bw_util 0.70% |
| V1 | vectorized 128-bit paged gather | mean ~196 µs (4–10.8× over V0) | LD count ~8× lower; 543 GB/s big shape |
| V2 | drop `.tolist()` sync + enable cudagraph | eager 100.7 / cudagraph 67.0 µs | 1.95× eager |
| V3 | `data_ptr`-keyed launch cache | eager 66.6 ≈ cudagraph 67.1 µs | host gap eliminated |
| V4 | cp.async 2-stage double-buffer | 47.1 / 47.4 µs (1.41×) | BW 543→1192 GB/s; tflops 71.8; trtllm gap 3.45→2.44× |
| **V5** | **split-KV + combine + cp.async (warp-MMA anchor)** | **45.1 / 45.2 µs** | 31/31, rel 0.00213; 2.8× vs vllm FA varlen 127.6 µs |

### tcgen05 rewrite + productionization (cudagraph GPU-only)

| Stage | Technique | Latency | Note |
|---|---|---|---|
| V9 | tcgen05 QK^T→TMEM→softmax foundation | single-tile 8.8→11.3 µs | launch-bound; 3.2× vs scalar 28.5 µs |
| V10 | in-kernel P@V fusion cracked | single-tile 11.2 µs | correctness breakthrough, rel_l2 0.0018 |
| stage4 | first integrated tcgen05 split-KV decode | mean 149.6 µs (max 398.9) | end-to-end baseline for the rewrite |
| a3 | KSTAGES=2 + drop `sQn` (64 KB) | shape0 decode 30.6 µs (1.9×) | first to beat stage4's 43 µs on shape0 |
| a4 | warp-spec + split-KV (NS=4) + combine | shape0 18.56 µs; mean 47.71 µs | approaches trtllm |
| a4 + bf16 partials | halve combine read BW | **mean 44.11 µs** (min 15.55 / max 104.84) | beats V5 45.3 µs; 3.4× vs stage4 |
| combine (N_DCHUNK=32) | occupancy knob | combine 26.65→10.34 µs | full pipeline 53.31→43.15 µs |

Baselines (cudagraph GPU-only): vllm FA varlen 127.6 µs; trtllm-gen 19.3 µs (early record; cudagraph mean 19.6 µs) refined to **12.3 µs full-path (fmha body 9.3 µs)** on an idle GPU. trtllm remains ~1.5× ahead on shape0; the residual gap is per-CTA warp count (16 warp vs 4 warp), not occupancy.

## Anti-Patterns

| Don't | Because |
|---|---|
| Tune `block_n`/tile size to raise occupancy when grid < 148 SMs | occupancy is GRID-limited; `block_n=32` regressed to 46.2 µs, `block_n=128` overflowed SMEM opt-in (V6) |
| Expect a Triton auto-tune to reach trtllm | Triton floors at 26.2–26.3 µs; large `page256` shapes 42–46 µs — needs hand-written tcgen05 (V8) |
| constexpr-unroll a 4–16 tile KV loop with a heavy tcgen05 body in one CTA | blows the LLVM module ("Failed creating llvm::Module"); use split-KV + `range_constexpr` specialization |
| Put a `make_tmem_copy` atom inside a dynamic `cutlass.range` loop | cannot legalize in dynamic `scf.for`; specialize by `num_tiles` (only 4/16 exist) with `range_constexpr` |
| Reuse the FA4 hd256 forward for decode | it is prefill-only (no SplitKV), ~4× slower (klen=1024 ~400 µs vs 104 µs) |
| Chase within-CTA warp overlap or N=128 tiles on the mean-killer shapes | O accumulator is a serial chain; a6 +10%, a7 slightly slower — latency-bound, less work ≠ less exposed latency |
| Add a 2nd CTA/SM at D=256 to lift occupancy | O alone owns 256 of 512 TMEM cols → two CTAs' O won't fit (harder than the SMEM wall) |
| Vectorize Q/O loads or drop rescale to fight latency | ≈0% / −20% (doubled fixed cost); only more concurrent warps hide TMEM-read latency |
| Build tensor-TMA paged-KV from a minimal PoC | the tensor-descriptor path device-hangs standalone; `cp.async.bulk` works and load was never the bottleneck |
| Trust eager `do_bench` wall-clock | host launch dominates (~112 µs rebuild + `.item()` sync); judge with cudagraph/GPU-only |
| Report a single best shape (18.5 µs) | that is a klen=256 small-B special case; the true 31-shape mean is ~44 µs |

## Further Reading

- **Pitfalls (traps → symptom → cause → fix):**
  [`docs/pitfalls/nvidia/cutedsl/sm103-paged-attention-decode-pitfalls.md`](../../../../pitfalls/nvidia/cutedsl/sm103-paged-attention-decode-pitfalls.md)
- **Full v0→v10 + warp-spec journey (version ladder + per-version detail):**
  [`docs/ref-docs/nvidia/cutedsl/sm103/sm103-paged-attention-decode-optimization.md`](../../../../ref-docs/nvidia/cutedsl/sm103/sm103-paged-attention-decode-optimization.md)
