# Flash Attention on sm90 — Optimization Journey (ref-docs)

**Last Updated**: 2026-07-03  ·  1 recorded attempt(s)

Hardware: H100 / H200 / B200 / B300 / Hopper / Blackwell / sm90 / sm90a / sm100 / sm103 · DSL: vLLM v1 attention backend (fa_utils.py) · dtype: n/a (dispatch-logic finding) · shapes: gated on head_size

---

## Attempts in detail

### ➖ version selection fallbacks

工况: vLLM v1 attention backend 在 Hopper/Blackwell 上运行时自动选择 FA2/FA3/FA4,逻辑集中在 vllm/v1/attention/backends/fa_utils.py:get_flash_attn_version()(vllm HEAD 93d8f834d)。
方法: 通读版本选择与降级分支,厘清何时真正启用 FA4。
预期: 精确判断某卡/某 head_size 上实际生效的 FA backend。
实测: 默认 SM90→FA3、SM100(major==10 且 is_fa_version_supported(4))→FA4、其余→FA2;FA4 会被强制降级回 FA2 的条件有——head_size>128 且 ≠192(TMEM 容量,Dao-AILab #1959)、VLLM_BATCH_INVARIANT 开启(破坏 batch invariance)、ALiBi、以及 MLA 非标准 paged layout(CuTeDSL 不支持)。
归因与结论: 中性/参考。要点: 在 Blackwell 上'硬件支持 FA4'不代表实际走 FA4——head_dim(尤其 256)、batch-invariant 开关、ALiBi、MLA layout 任一命中都会静默 fallback 到 FA2;调试 attention 性能/数值前应先确认 get_flash_attn_version() 在当前 shape+config 下实际返回的版本。

<sub>`x-backend_selection` `x-tmem_capacity`  session `89c40dec` · commit `93d8f834d`</sub>

