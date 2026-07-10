# Flash Attention Prefill on sm103 — Pitfalls

Traps hit while optimizing `flash_attention_prefill` — vLLM (pai b46dc08) ModelOpt NVFP4 path + FA4 attention, NVFP4 MoE weights + bf16 attention, qwen3.7-max / qwen3.5-plus (NVFP4 MoE, hd256), TP4/TP8 on B300 / Blackwell / sm103.
Companion to:

- Optimization journey: [journey](../../../../ref-docs/nvidia/common/sm103/flash_attention_prefill-optimization.md)

---

## 1. vllm modelopt nvfp4 gotcha

**Trap**: 为验证 FA4 需先把 NVFP4 MoE 模型(qwen3.7-max/3.5-plus)在 pai-vllm 拉起;遇到 weight-load 崩溃与环境变量非法值,逐一定位修复（预期 模型正常加载,进入 attention 前向以测试 FA4 hd256）

**Result**: 两个 bringup blocker:(1) 环境里 VLLM_NVFP4_GEMM_BACKEND=atrex 是 b46dc08 的非法值(合法:flashinfer-cutlass/trtllm/cudnn/cutlass)→ 崩;它与 VLLM_NVFP4_USE_ATREX 开关是两回事,须显式传 flashinfer-cutlass。(2) vLLM modelopt.py:304-311 只读 checkpoint 的 exclude_modules→回退 ignore(121 条,不含 shared_expert),完全无视 HF 标准字段 modules_to_not_convert(664 条,明确列出所有 shared_expert.{gate,up,down}_proj 应不量化)→ 把本该 BF16 的 shared-expert 误建成 NVFP4(uint8)与 checkpoint 冲突 → load 崩。修法:让 modelopt config 合并 modules_to_not_convert 进 exclude 列表

**Why**: 生产用的 vLLM(f19c3221)+atrex 能处理这些,但本 checkout b46dc08 不能:setdefault 让 shell 全局的非法 GEMM backend 泄漏进来;modelopt 用了不完整的 ignore 列表漏掉 shared_expert。都是 pai-vllm 与生产版本差异导致的集成坑,非 FA4 内核本身

**Lesson**: 负向(集成坑,非 FA4 内核问题),但可复用:在非生产 vLLM checkout 上跑 NVFP4 模型时,(a) 别信 shell 里遗留的 backend 环境变量,显式指定合法 GEMM backend;(b) 若 weight-load 报量化冲突,查 modelopt 是否漏读 HF 的 modules_to_not_convert(常只读了 ignore/exclude_modules 的子集)。这类版本差异会在真正测到 FA4 attention 之前就拦住,须先扫清才能定位内核问题。

<sub>`quantization` `x-build-integration` `x-build-integration`  session `4b8ff1d7`</sub>

