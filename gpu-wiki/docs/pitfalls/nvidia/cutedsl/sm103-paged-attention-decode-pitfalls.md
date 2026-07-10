# sm_103 CuTeDSL paged-attention-decode pitfalls

Traps hit while taking a `paged_attention_decode` kernel from a warp-MMA baseline
to a `tcgen05`/TMEM warp-specialized decode on **NVIDIA B300 / sm_103a (Blackwell
Ultra, HBM3e, 148 SMs, TMEM 128 lane × 512 fp32 col)** under **CuTeDSL cutlass
4.5.2**. Workload: bf16 in/out, fp32 softmax, `hd256`, GQA `q_per_seq=4`,
`block_size ∈ {64,256}`, `block_n=64`, `klen ∈ {256, 1024}`. Traps 1–9 are the
dead-ends; traps 10–14 are silent-correctness bugs found along the winning path.

## 1. constexpr-unrolled KV loop blows the LLVM module

### symptom

A single-CTA kernel that processes all KV tiles (D=256, up to 4 tiles: GQA pack +
paged gather + V transpose + causal-tail + multi-tile online softmax + TMEM-O
rescale) passes the MLIR stage but fails with
`Failed creating llvm::Module / serializing the module` (sm_103a lowering).

### cause

CuTeDSL `constexpr` loops **fully unroll the IR**. A heavy `tcgen05` body
(MMA + TMEM ops) times 4 tiles exceeds the LLVM module serialization limit.
Bisect confirms it: forcing `num_tiles=1` compiles and runs (all other logic is
fine), so the unroll — not any single component — is the cause.

### fix

Use **split-KV** so each CTA handles only 1–2 tiles: the constexpr unroll shrinks
naturally and the grid widens (`B·nkv → B·nkv·num_splits`, higher occupancy),
reusing the V5 combine kernel. Split-KV here is not just a perf optimization — it
is the **prerequisite for the integrated kernel to compile at all**. (The
alternative, a dynamic `cutlass.range` carrying online-softmax state, hits trap #2.)

## 2. `make_tmem_copy` cannot legalize inside a dynamic `cutlass.range` loop

### symptom

Trying to run 16 KV tiles in one CTA with `cutlass.range(unroll=1)` (dynamic, body
compiled once) + loop-carried `run_max`/`run_sum`/`O_acc` and a `tcgen05`
`make_tmem_copy` inside the loop fails with `MLIRError` / ICE:
`builtin.unrealized_conversion_cast: atom.tmem_load → tiled_copy remained live`
inside the dynamic `scf.for` region. Every variant fails — atom built inside or
outside the loop, inlined `get_slice`, crossing a function boundary, runtime vs
constant loop bounds. Attempting the same with loop-carried constants also hits
`src is structured different after this for` (state is constexpr at entry, dynamic
inside; `+0` gets folded; tensorizing `rMax`/`rSum` still fails loop-carry syntax).

### cause

In cutlass 4.5.2 the tmem-copy atom produced by `make_tmem_copy` **cannot be
legalized across a dynamic `scf.for` region boundary** — the same root cause as
the warp-spec `scf.if` limit (trap in #11 territory: atoms cannot cross scf
regions). FA4 gets away with dynamic loops only via its full warp-specialized
machinery.

### fix

CuTeDSL already specializes per integer constant, and production `klen` has only
two values (4/16 tiles). Use **`range_constexpr` specialized by `num_tiles`** — a
lean per-tile body keeps even the 16-tile constexpr from blowing the IR (verified:
D=64 16-tile `rel=0.0013` PASS, D=256 16-tile PASS, ~few-minute compile). Do **not**
put a TMEM-copy-bearing loop under a dynamic `cutlass.range` in a plain
128-thread kernel.

## 3. `block_n` / tile-size tuning cannot move grid-limited occupancy

### symptom

On the V5 split-KV kernel (45.1µs), tuning `block_n=32` regresses to 46.2µs (tile
count doubles; per-tile overhead exceeds any occupancy gain; big shapes 60→75µs),
and `block_n=128` throws `CUDA_ERROR_INVALID_VALUE` (2×128×256×2B/stage exceeds
the SMEM opt-in). Occupancy does not move at ~6.2% / 254 regs.

### cause

On the large non-split `page256` shapes, `grid = 64 < 148 SMs`. Occupancy is
**GRID-limited, not SMEM/register-limited** — with only 64 blocks, a second block
per SM is physically impossible, so no tile/stage/register knob can raise
occupancy. V5 is the actual ceiling of the warp-MMA architecture; the remaining
2.33× to trtllm needs a structural MMA-path change.

### fix

When `grid_CTAs < num_SMs`, compute grid/SM first — occupancy is fixed by the
grid. Either **split-KV to add CTAs**, or switch to the **tcgen05/UMMA+TMEM path**
that moves the register-resident `O[M,256]` out of registers. Parameter tuning is
wasted effort in this regime.

## 4. within-CTA warp overlap cannot shorten the serial O-accumulator chain

### symptom

Two full new kernels aiming for mean <30µs both come out correct but **not faster**:
a6 (FA4-style 10-warp pipeline: dedicated mma/softmax/correction/load warps,
S/P TMEM double-buffer, correction via `sScale` smem broadcast) is shape9 decode
100µs vs a4's 91µs (**+10%**); a7 (N=128 / Kv128, halving tile count) is shape9
105 vs 104µs and shape0 22.4 vs 17.3µs (slightly slower).

### cause

The **O accumulator is a serial dependency chain** — each tile's PV needs the
previous tile's O already rescaled. Adding warps or pipeline stages cannot shorten
this chain, only add barrier cost. And because the kernel is latency-bound,
reducing work (a7's bigger MMA + 2-half PV + 2-block gather) does not reduce
unhidden latency.

### fix

For a kernel whose output accumulator has a serial dependency, within-CTA
multi-warp/pipeline overlap does **not** speed it up — it only adds a barrier tax.
Redirect effort to more concurrent CTAs/warps (split-KV). Two paired traps here:
(a6) an `elect_one` placed inside `if warp_idx < SOFTMAX_WARPS` makes each of 4
warps arrive once on a count-1 barrier → parity flips back → the mma's wait never
satisfies → **deadlock**; arrive exactly once. (a7) with `block_size=64` but
`N=128`, a tile spans two physical pages — gather each 64-key sub-block from its
own `block_table` entry or keys 64–127 read adjacent garbage (symptom:
QK/`run_max` already wrong, changing P/V has no effect).

## 5. latency-bound + 1 CTA/SM: work-reduction and vectorization are all no-ops

### symptom

On the mean-killer shape9 (klen1024, B56, NS=8, ~89.8µs): `waves/SM=3.03`
(occupancy is fine), but `sm_throughput 7.28%`, `warps_active 7.78%` (vs trtllm's
21%, ~3× headroom), and `long_scoreboard` = 61% of stall cycles. Levers measured:
within-CTA pipeline −10%; `tps=1` (removing all rescale) −20% but fixed per-CTA
cost doubles and cancels it; vectorizing Q/O ≈0%; skip Q-gather saves 3.4µs, skip
O-write saves 8.6µs.

### cause

Decode is **latency-bound**: the compute warp's TMEM traffic (S read + O rescale
read/write + O readback) is exposed with only **5 warp / 1 CTA per SM** — no other
warp covers the wait. All three latency-hiding routes are blocked by hard
resources: (1) 2 CTA/SM is impossible — not only SMEM (`sQ64+sK+sV=128KB>116KB`)
but **TMEM**: one CTA's O owns 256 of 512 columns, so two CTAs' O (512 cols) won't
fit — even shrinking M=64 to save SMEM can't help; (2) batching 4 O D-chunks to
hide latency needs 256 reg/thread (currently 108) → register blowup; (3)
vectorization only cuts instruction count, not latency.

### fix

Under latency-bound + 1 CTA/SM, work-reduction / vectorization / rescale-removal
are all ineffective — **only more concurrent warps hide the latency** (this is why
the next redirect is 16-warp warp-spec, trap #7 / see ref-docs). When judging the
occupancy ceiling, check **both** SMEM and TMEM: at D=256, O owning half the TMEM
is a harder 2-CTA/SM wall than SMEM.

## 6. tensor-TMA descriptor path device-hangs standalone

### symptom

De-risking a trtllm-level rewrite by building a from-scratch TMA paged-KV tile-load
PoC (~30 variants: plain/swizzled layout, `make_tiled_tma_atom_B`, quack
`tma_get_copy_fn`, 2D/4D gmem, `PipelineTmaAsync`, cluster launch, stride-alignment
marks, D=128/256, two envs). **Every tensor-TMA copy variant device-hangs** — even
fire-and-forget with no wait, so the copy instruction itself device-faults.
Isolation: `expect_tx` without a copy completes cleanly; the FA4 forward runs in
the same env (CC 10.3, 0.045ms), proving TMA hardware works; a plain
`cp.async.bulk` (non-tensor, no descriptor) standalone succeeds (`max_err=0`).

### cause

The fault is precisely in the **tensor-descriptor path (`make_tiled_tma_atom_B`)**,
not the async-bulk mechanism, not mbarrier/`expect_tx`/wait, not the env. A
from-scratch tensor-TMA only works inside FA4's complete persistent/warp-spec
kernel context — it cannot be reproduced piecemeal.
(`compute-sanitizer` was unusable: xingze env CUDA runtime 13.3 vs driver 13.2
mismatch cannot attach; system env cannot report a hung kernel.)

### fix

Do not grind on tensor-TMA descriptors from a minimal PoC. Validate the async-bulk
path with `cp.async.bulk` first; if you need tensor-TMA, adopt/reuse a complete
runnable reference kernel rather than assembling a minimal PoC bit by bit.
Crucially, **load was never the real bottleneck** — a plain bulk-copy of
re-paged-128 contiguous KV pages suffices; the real bottleneck is the compute-warp
structure (M=128 mostly padding + 4 warps that cannot hide TMEM-read latency).
Diagnostic: a device-hang where even fire-and-forget hangs = the copy instruction
itself is faulting.

## 7. Triton auto-tuning has a hard ceiling on SM103

### symptom

Exhaustive Triton heuristic tuning (split sweep sp=1..4, `num_warps`,
`packed_base` thresholds) bottoms out at 26.2–26.3µs (shapes 16,17 improve
3.8–5.9µs; shape16 sp=4/nw=4=27.9 vs sp=3/nw=8=31.6). But the large rep=16
`page256` shapes (7–9, 28–30) stay stuck at 42–46µs, 254 regs, ~6% occupancy,
instruction-throughput-limited — `BLOCK_N`/stages/warps/splits move nothing.

### cause

The large `page256` shapes have hit the Triton-on-SM103 ceiling. Closing the gap
to trtllm (19.3µs) needs hand-tuned `tcgen05` scheduling/TMA: Triton auto-emits
`tcgen05` instructions but cannot reach hand-written CUDA quality.

### fix

Accept that an auto-compiler's tuning space has a hard upper bound; past it you
must go to a hand-written `tcgen05` path. Silver lining: the same `tcgen05` blocks
(`MmaF16BF16Op(64,64,16)` + TMEM accumulator + `Ld16x64bOp` readback) were proven
to compile and run on SM103 here, paving the CuTeDSL rewrite (V9/V10).

## 8. FA4 hd256 forward is prefill-only — not a decode kernel

### symptom

Reusing FA4's hd256 2-CTA `flash_attn_varlen_func` (+ `page_table`, re-paged to
`page_size=128`) for the 31-shape decode is **correct but slow**: `rel=0.0021`
PASS, but klen=1024 ~400µs vs a4's 104µs (~4× slower); `num_splits>1` errors (no
SplitKV); `pack_gqa` makes no difference (398 vs 404µs). (`seqused_k` unsupported,
but decode KV is uniform so it can be dropped with `max_seqlen_k=klen`.)

### cause

The FA4 hd256 kernel is a **prefill kernel** — no SplitKV, no decode (small
`q_len`) optimization — so on `q_len=4` decode it is grid-underutilized and lacks
flash-decoding. There is no off-the-shelf D=256 *decode*-optimized kernel to reuse.

### fix

Before reusing an attention kernel, confirm whether it is **prefill- or
decode-shaped**: a prefill kernel lacks SplitKV and will be several times slower on
tiny-`q_len` decode. Correct ≠ fast. A <30µs decode must be built from scratch as
a trtllm-level kernel.

## 9. CuTeDSL 4.5.2 compile-time legalization gotchas (checklist)

### symptom

Repeated minutes-long failed compiles across the double-buffer pipeline (V4) and
tcgen05 integration (Stage 3c): `range_constexpr requires constexpr`, compile-time
values silently becoming runtime, swizzled-tensor write failures, flatten/closure
errors, atom-mismatch on TMEM copies.

### cause

Each is a CuTeDSL layout/legalization constraint that is invisible until you hit it.

### fix

Follow this checklist when writing tcgen05 kernels:

1. A `@cute.kernel` body does **not** close over `@cute.jit` host locals — pass every
   compile-time value as a `Constexpr`/typed kernel arg (else `range_constexpr
   requires constexpr`, or the value becomes runtime).
2. Use `range_constexpr` in **statement form** for compile-time loops — never inside a
   comprehension.
3. Encode a double-buffer stage as a **runtime tensor MODE** (staged layout, indexed
   `[...,stage]`), not a per-stage Python list.
4. Swizzled `sQ`/`sV` cannot be written scalar/plain — write a plain `sQn`/`sVn` first,
   then `relayout` (`autovec_copy`) into the swizzled MMA tensor.
5. Coord/identity tensors cannot be flattened via `make_tensor(coord_iterator)` —
   index linearly (`tScS[i]`).
6. Closures capturing a tensor are unsupported inside dynamic control flow — inline
   the expression.
7. `make_tmem_copy` must act on a `flat_divide` 2D epi-tile slice
   (`tCtS[(None,None),0,0]`) and needs a clean 128-row fragment
   (`make_trivial_tiled_mma`; M=64 gives a composite `(16,4)` that won't fit).
8. `autovec_copy` requires equal source/dest bit-width (`fp32→bf16` needs `.to(bf16)`
   first); 16-bit cp.async is unsupported (only 32/64/128).

Main line: compile-time values go through `Constexpr` args; swizzled tensors go
through a plain staging buffer + relayout; TMEM copies act only on clean 2D slices.

## 10. SMEM opt-in request below actual use = silent runtime OOB

### symptom

A warp-spec kernel with KSTAGES=2 crashes at runtime with `CUDA illegal memory
access` — but **compiles clean**. Setting the launch SMEM request to 228000 (below
the ~229376 actually used) triggers it.

### cause

On Blackwell, a launch SMEM request slightly smaller than the real usage does not
fail to compile; it faults only at runtime (silent OOB).

### fix

Push the SMEM request to `cudaDevAttrMaxSharedMemoryPerBlockOptin` (232448 on this
B300). When you see an unexplained illegal-access, check the SMEM request value
first.

## 11. split-dimension knob must exactly divide the split dimension

### symptom

The combine kernel's `N_DCHUNK=48` produces silently wrong output on some columns.

### cause

`48 → DC=5` covers only 240 of D=256 columns — a silent tail-drop, no error.

### fix

Any knob that splits a dimension (`N_DCHUNK` over D=256) **must divide it exactly**;
use `N_DCHUNK=32` (DC=8). Same class applies to any tile/chunk divisor.

## 12. tcgen05 readback fragment must be built from the destination shape

### symptom

A Stage4 split-KV kernel gives `rel_l2 = 1.000000` (PV→O reads back all zeros),
while the single-CTA stage3a reference still passes at 21µs. Earlier probes
(entry-write / gOp scalar-write) all looked "bound" or "PV=0".

### cause

The O readback register tensor was created from the **source (TMEM) partition
shape** instead of the **destination (gmem) partition shape**, so the `Ld16x64`
epi-copy silently reads back 0. Every upstream probe was overwritten by the
readback (which fills all of `mOp[0,0,0]`), masquerading as "binding broken / PV=0".

### fix

Build the tcgen05 readback fragment from the **destination** tensor shape, never the
source. When output is all-zero/all-wrong, first confirm the upstream truth with an
**independent read path** (here, an `Ld32x32` per-row read of TMEM_O gave
`sumsq=32.14`, proving PV was always correct), then isolate segment by segment —
don't let a buffer-overwriting readback mislead the diagnosis.

## 13. PV computes P@(sV)^T, and all-masked rows produce NaN

### symptom

Two coupled numeric bugs on the PV/O path: (a) with natural V loaded, `O = P@V^T`
(`rel = 1.39` / up to 1.0), not attention's P@V; (b) padding rows (`m_idx ≥ m_rows`)
are all masked → `tile_max = -inf` → `exp2(-inf − (−inf)) = NaN`.

### cause

(a) tcgen05 PV MMA computes `P @ (sV)^T` **regardless of `b_major`**; orientation is
set by `sV`'s physical layout, and `autovec_copy` with the same TV layout is an
identity copy (no transpose). (b) an all-`-inf` row max feeds `-inf − (−inf)` into
`exp2`.

### fix

Store `sV` **physically transposed** (`sVn_T` stride-swapped source view,
`ldmatrix.trans`, or TMA — cp.async cannot transpose) so PV yields P@V. Clamp the
row max with `fmax(tile_max, -1e30)` so masked rows give 0, not NaN. Resolve
orientation with a deterministic value probe (`V[n,d]=d+1`, P selecting key 0), not
blind `b_major` sweeps.

## 14. warp-spec overwrite-class races (aliased SMEM, wrong tiling)

### symptom

Uniform ~0.245 error across **all** queries/heads (an overwrite signature); or
`nonzero_rows=112` (K/Q under-loaded).

### cause

(a) SMEM aliasing — `sVn` reusing `sQn` — lets the load warp's tile-0 V-gather
overwrite Q while the compute warp is still reading it. (b) The load-warp cp.async
was tiled by the full 160 threads instead of its real 32, so lanes don't cover the
data.

### fix

Under warp-spec, **barrier any aliased SMEM** and move Q-pack ahead of the
load/compute split (`sync_threads()`); tile the load-warp copy by its **real 32
threads** with `lane = tidx%32`. Diagnostic: if the error is *uniform across all
rows*, suspect an overwrite/race, not the math.

## evidence + reproduction

- Warp-MMA path (V0–V8) and the tcgen05 foundation (V9/V10): session
  `6270b3b2-1ef0-4841-b0c9-34935a8663fa`, commits `0b7511f` (V0), `14ea770` (V1),
  `17b7f90` (V3), `2346ae4` (V4), `9a14c6d` (V5), `8e5b080` (V6), `3bfb22d` (V7),
  `ac54f15` (V9). Traps #1, #3, #7, #9 reproduce there.
- Warp-spec productionization (a0–a7, combine, TMA PoC): session
  `3e24042c-d362-4a41-bc2a-cdae97c5640e`, commits `f68fb22` (a3/a4/combine),
  `2b76d06` (combine / readback fix). Traps #2, #4, #5, #6, #8, #10–#14 reproduce
  there.
- Bisect for trap #1: force `num_tiles=1` → compiles and runs. For trap #12: use an
  independent `Ld32x32` per-row TMEM read to confirm upstream truth. For trap #6:
  `cp.async.bulk` standalone (`max_err=0`) isolates the fault to the tensor-descriptor
  path.
- Correctness held throughout: 31/31 shapes PASS, `rel ≈ 0.0021` (bf16 partials
  0.0029).

## affected versions

- CuTeDSL cutlass 4.5.2 (two Python installs both 4.5.2 — confirmed a code-structure
  issue, not a version delta)
- sm_103 / sm_103a lowering, NVIDIA B300 (Blackwell Ultra), CC 10.3
- HBM3e, 148 SMs, TMEM 128 lane × 512 fp32 col, SMEM max opt-in 232448 B
- Baselines compared: flashinfer trtllm-gen (`fmhaSm100 H256 PagedKv Causal P64 Q16
  Kv128 Persistent`), FA4 `flash_attn_varlen_func` hd256, vLLM FA varlen
- Related: [`sm103-paged-attention-decode` kernel-opt](../../../kernel-opt/nvidia/cutedsl/sm103/sm103-paged-attention-decode.md),
  [full optimization journey](../../../ref-docs/nvidia/cutedsl/sm103/sm103-paged-attention-decode-optimization.md)
