# Blackwell Ultra (SM103 / B300) CuTeDSL Optimization Journeys

Full per-kernel optimization journeys on the Blackwell Ultra (sm_103, B300) architecture.

---

| File | Description |
|------|------|
| [sm103-paged-attention-decode-optimization.md](sm103-paged-attention-decode-optimization.md) | CuTeDSL paged-attention decode v0→v10: scalar gather → vectorized 128-bit gather → cp.async pipeline → split-KV + combine → tcgen05 QK/P@V fusion; warp-MMA anchor 45.1 µs, tcgen05 path to ~44 µs (bf16 partials) |
| [sm103-fa4-hd256-prefill-optimization.md](sm103-fa4-hd256-prefill-optimization.md) | FA4 hd256 prefill: v0 dense 2-CTA baseline → exp2 frequency tuning → v2 SM103 hardware SFU exp2; trtllm-gen performance-ceiling reference |
