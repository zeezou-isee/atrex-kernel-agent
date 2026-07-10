# Flash Attention on sm100 — Pitfalls

Traps hit while optimizing `flash_attention` — CuTeDSL, pulled by vLLM CMake, n/a (provenance/build finding), n/a on B200 / B300 / Blackwell / sm100 / sm100a / sm103.
Companion to:

- Optimization journey: [journey](../../../../ref-docs/nvidia/cutedsl/sm100/flash_attention-optimization.md)

---

## 1. hd256 seqused 2cta constraint

**Trap**: 核实 FA4 在 Blackwell head_dim=256 专用 2CTA kernel 路径上对 seqused 的支持（预期 确认 hd256 + paged-KV 是否能带 seqused 在 FA4 上跑）

**Result**: 不能。pin commit interface.py:978-979 硬断言: SM100 head_dim=256 dedicated 2CTA kernel 路径下 assert seqused_q is None and seqused_k is None(报错 'SM100 forward with head_dim=256 does not support seqused_q/seqused_k');同路径还禁用 softcap / block sparsity / learnable_sink。这正是 vllm fa_utils.py 把 SM100 上 head_size>128 且 ≠192 直接降级到 FA2 的根因——hd256 无法在 FA4 上带 paged-KV 的 seqused 运行

**Why**: hd256 因 TMEM 容量需走 2-CTA 专用 kernel,该 kernel 未实现 seqused;缺 seqused 时 paged-KV padding 未初始化会引发数值问题(对应 memory 的 fa4-hd256-seqused-padding-nan-bug),故上游直接禁用而非静默产错

**Lesson**: 负向/约束。根因: hd256 因 TMEM 容量必须走 2-CTA 专用 kernel,而该 kernel 未实现 seqused;若强行不带 seqused,paged-KV 的未初始化 padding 会在 P@V 里产生 0*NaN(对应 memory fa4-hd256-seqused-padding-nan-bug),故上游选择直接断言禁用而非静默出错。可操作规则: 在 Blackwell 上做 hd256 paged-attention 且需要 seqused/变长有效长度时,不能指望官方 FA4——要么落到 FA2、要么自研 hd256 kernel 并显式处理 padding;'FA4 支持 varlen+seqused'这一整体结论在 hd256 2CTA 路径上不成立。

<sub>`tmem` `cta_cooperation` `x-tmem_capacity` `numerical_instability`  session `89c40dec` · commit `2c839c33742309ec41e620bf837495ec9926c56e`</sub>

