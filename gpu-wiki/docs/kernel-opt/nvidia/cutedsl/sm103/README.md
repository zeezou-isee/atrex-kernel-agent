# SM103 Blackwell Ultra (B300) CuTeDSL

CuTeDSL attention-kernel optimization highlights for the Blackwell Ultra (sm_103, B300) architecture.

> Full version-by-version journeys: [ref-docs/nvidia/cutedsl/sm103/](../../../../ref-docs/nvidia/cutedsl/sm103/). Pitfalls live in [pitfalls/nvidia/cutedsl/](../../../../pitfalls/nvidia/cutedsl/) (files prefixed `sm103-`).

---

| File | Description |
|------|------|
| [sm103-paged-attention-decode.md](sm103-paged-attention-decode.md) | Paged-attention decode highlights: vectorized 128-bit paged gather, cp.async multi-stage KV pipeline, split-KV occupancy (beats within-CTA overlap), tcgen05 QK→TMEM softmax, in-kernel P@V fusion |
| [sm103-fa4-hd256-prefill.md](sm103-fa4-hd256-prefill.md) | FA4 hd256 prefill: serving-correct 1-CTA path (fixes the 2-CTA TMEM padding-NaN garbage) + dense 2-CTA exp2 / tile-n tuning |
