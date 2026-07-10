# flashinfer trtllm-gen paged-decode baseline pitfalls on sm_103 (B300)

_Last Updated: 2026-07-02_

Traps hit while reproducing the flashinfer **trtllm-gen** (NVIDIA prebuilt
cubin) paged-decode baseline on B300 / sm_103. The baseline is a thin wrapper
(`trtllm_fixed.py`) over flashinfer's trtllm-gen decode kernel; workload is the
decode stage over 31 production shapes (bf16, hd256, GQA, `q_per_seq=4`, block
size 64 / 256, with the large shapes at block256 / klen1024).

Both traps are environment / toolchain issues, not algorithmic ones: the first
prevents the baseline from running at all on the wrong flashinfer build, and the
second makes even the fully-running baseline report host latency instead of
kernel time.

## 1. cubin / flashinfer version mismatch — `loadKernel` failures on the wrong build

### symptom

Running `trtllm_fixed.py` on the already-installed flashinfer **0.6.9**:

- 15 of 31 shapes fail with `loadKernel` errors — concentrated on the large
  block256 / klen1024 shapes.
- The shapes that *do* run measure ~55 µs.
- Setting `FI_FORCE_JIT=1` does nothing.

### cause

trtllm-gen kernels ship as **NVIDIA prebuilt cubin** — they cannot be JIT-built
from source, so `FI_FORCE_JIT` is useless here. Each cubin is strongly bound to a
specific flashinfer build version and shape configuration: a `loadKernel`
failure means the installed cubin simply does not contain that shape's
configuration.

`trtllm_fixed.py` was written against beiyuan's particular flashinfer set. The
from-source build (commit `0cb2bc9`) does not resolve it either — its backend
FFI signature does not match the 0.6.9 backend.

### fix

Upgrade the flashinfer build to the one the wrapper matches — **0.6.12** makes
all 31 / 31 shapes pass. Concretely:

- To reproduce a trtllm-gen baseline, lock to the flashinfer build that matches
  it; do not try to patch the wrapper.
- Treat a `loadKernel` failure as "cubin is missing this shape config" → upgrade
  or swap the flashinfer build, not edit code.
- `FI_FORCE_JIT` cannot rescue a prebuilt-cubin kernel — do not reach for it.
- When porting a thin wrapper like `trtllm_fixed.py` across environments, align
  the FFI signature **and** the cubin version together.

## 2. `seqused_k.max().item()` host-sync makes the baseline launch-bound

### symptom

Per-shape latency is nearly **constant at ~55-68 µs and does not scale with KV
length**. The recorded (beiyuan) trtllm-gen baseline is ~19.3 µs mean, so the
local numbers are 3×+ too high. Even after flashinfer 0.6.12 fixes the
`loadKernel` failures (31 / 31 passing, trap 1), latency is still ~62 µs rather
than ~19 µs.

### cause

`trtllm_fixed.flash_decode` calls `seqused_k.max().item()` on every invocation.
`.item()` forces a GPU→CPU synchronization each call, making the measured path
host / launch bound. A latency that stays flat and does not grow with KV length
is the classic launch-bound signature — it is host latency, not the GPU kernel's
real execution time.

### fix

- When evaluating any decode baseline, first check whether latency scales with
  `seqlen`. If it is flat, suspect a host sync / launch overhead before trusting
  the number.
- Remove implicit synchronizations such as `.item()` / `.max().item()` from the
  timing path (precompute or pass the value some other way), otherwise you are
  measuring host latency instead of kernel performance.

## evidence + reproduction

- Hardware: B300 / sm_103, Blackwell.
- Workload: paged-decode baseline via flashinfer trtllm-gen prebuilt cubin,
  bf16, 31 production shapes (hd256, GQA, `q_per_seq=4`, block 64 / 256; large
  shapes block256 / klen1024).
- Trap 1: flashinfer 0.6.9 → 15/31 `loadKernel` failures, runnable shapes
  ~55 µs, `FI_FORCE_JIT=1` ineffective; from-source build `0cb2bc9` FFI mismatch;
  flashinfer 0.6.12 → 31/31 pass.
- Trap 2: latency flat at ~55-68 µs (independent of KV length); ~62 µs on 0.6.12
  vs recorded ~19.3 µs mean; root cause `seqused_k.max().item()` GPU→CPU sync in
  `trtllm_fixed.flash_decode`.
- Related sm_103 trtllm-gen / CuTeDSL paged-attention material lives under
  `../cutedsl/sm103/`.

## affected versions

- flashinfer 0.6.9 (`loadKernel` failures) and 0.6.12 (fixes them)
- from-source flashinfer build commit `0cb2bc9` (FFI signature mismatch)
- Blackwell B300, sm_103
