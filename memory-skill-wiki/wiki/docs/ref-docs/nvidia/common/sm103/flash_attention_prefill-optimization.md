# Flash Attention Prefill on sm103 — Optimization Journey (ref-docs)

**Last Updated**: 2026-07-03  ·  1 recorded attempt(s)

Hardware: B300 / Blackwell / sm103 · DSL: vLLM (pai b46dc08) ModelOpt NVFP4 path + FA4 attention · dtype: NVFP4 MoE weights + bf16 attention · shapes: qwen3.7-max / qwen3.5-plus (NVFP4 MoE, hd256), TP4/TP8

---

## Attempts in detail

### ❌ vllm modelopt nvfp4 gotcha

工况: B300/sm103,为测 FA4 需先在 pai-vllm(b46dc08)拉起 NVFP4 MoE 模型(qwen3.7-max/3.5-plus,hd256),TP4/TP8。
方法: 逐一定位阻塞模型加载的崩溃与环境变量问题。
预期: 模型正常加载并进入 attention 前向以测 FA4。
实测: 两个 bringup blocker。(1) 环境里 VLLM_NVFP4_GEMM_BACKEND=atrex 在 b46dc08 是非法值(合法仅 flashinfer-cutlass/trtllm/cudnn/cutlass),且它与 VLLM_NVFP4_USE_ATREX 开关是两码事——须显式传 flashinfer-cutlass,否则 setdefault 让 shell 全局非法值泄漏进来崩。(2) vLLM modelopt.py 只读 exclude_modules→回退不完整的 ignore(121 条,不含 shared_expert),无视 HF 标准的 modules_to_not_convert(664 条,明列所有 shared_expert 不量化)→ 把本该 BF16 的 shared-expert 误建成 NVFP4 → 与 checkpoint 冲突崩;修法是把 modules_to_not_convert 合并进 exclude。
归因与结论: 负向(集成坑,非 FA4 内核问题),但可复用:在非生产 vLLM checkout 上跑 NVFP4 模型时,(a) 别信 shell 里遗留的 backend 环境变量,显式指定合法 GEMM backend;(b) 若 weight-load 报量化冲突,查 modelopt 是否漏读 HF 的 modules_to_not_convert(常只读了 ignore/exclude_modules 的子集)。这类版本差异会在真正测到 FA4 attention 之前就拦住,须先扫清才能定位内核问题。

<sub>`quantization` `x-build-integration` `x-build-integration`  session `4b8ff1d7`</sub>

