# Flash Attention on sm100 — Optimization Journey (ref-docs)

**Last Updated**: 2026-07-03  ·  3 recorded attempt(s)

Hardware: B200 / B300 / Blackwell / sm100 / sm100a / sm103 · DSL: CuTeDSL, pulled by vLLM CMake · dtype: n/a (provenance/build finding) · shapes: n/a

---

## Attempts in detail

### ➖ provenance

工况: 调查 vllm-project/vllm(本地 clone HEAD 93d8f834d)里 FA4(CuTeDSL 版)的来源,目标是找到要修改 FA4 kernel 时该动哪份代码。
方法: 读 cmake/external_projects/vllm_flash_attn.cmake 的 pin 与安装 component,并核对 GitHub 上的 pin commit。
预期: 确认 FA4 CuTeDSL kernel 是否在 vllm 源码树内,以及其权威来源。
实测: FA4 CuTeDSL 实现(flash_attn/cute/)不在 vllm 主 tree 里;由 CMake 从 vllm 自己的 fork github.com/vllm-project/flash-attention.git(非 Dao-AILab 上游)pin commit 2c839c33742309ec41e620bf837495ec9926c56e 拉取,以 component _vllm_fa4_cutedsl_C 安装到 vllm/vllm_flash_attn/cute/;运行时 import 路径 vllm.vllm_flash_attn(未 build 环境 import 直接 ModuleNotFoundError)。
归因与结论: 中性/参考。要点: 在 vllm 源码树里 grep 不到 FA4 kernel 是预期行为——它是 build 时装入的外部 CuTeDSL component;若要读或改 vllm 的 FA4,必须去 vllm fork 的那个 pin commit(2c839c33),而不是 Dao-AILab 上游、也不是 vllm 主仓 tree;验证运行时实际加载的版本要查 env 里 vllm.vllm_flash_attn。

<sub>`x-external_dependency_pin` `x-build_provenance`  session `89c40dec` · commit `93d8f834d`</sub>

### ➖ varlen seqused support

工况: 核实 vllm 的 FA4(CuTeDSL,来自 vllm fork pin commit 2c839c33)对 varlen 与 seqused 的支持;FA4 在 SM100 走 fa_version=4 路径。
方法: 读 pin commit 的 flash_attn/cute/interface.py 与 vllm 侧 flash_attn_interface.py/flash_attn.py,定位公开入口、参数签名、校验与内部 varlen 判定。
预期: 确认 varlen/seqused 是否真被下发到 CuTeDSL kernel,而非仅 API 存在。
实测: 支持。公开 varlen 入口 flash_attn_varlen_func(interface.py:2353),底层 _flash_attn_fwd(:360);cu_seqlens_q/k、seqused_q/k 全在签名(:365-368),seqused_* 需为 (batch_size,) int32(:461-477);is_varlen 由 cu_seqlens 或 seqused 任一非空判定(:703-707);vllm 的 fa_version==4 分支调用 _flash_attn_fwd(..., seqused_k=seqused_k),backend 全程走 flash_attn_varlen_func;varlen+paged-KV 偏移约定 abs_q = q_idx + (seqlen_k - seqlen_q)(flash_attn.py:1236)。
归因与结论: 中性/参考。要点: FA4 的标准工作模式就是 varlen+paged-KV(非可选),seqused 是 varlen 的正规参数并有 int32/(batch,) 形状校验;移植或对齐 FA4 varlen 时,q/kv 索引用绝对偏移 abs_q = q_idx + (seqlen_k - seqlen_q)。注意此'整体支持'在 hd256 2CTA 专用 kernel 上有例外(见 fa4_vllm_hd256_seqused_2cta_constraint)。

<sub>`paged_gather` `x-feature_capability`  session `89c40dec` · commit `2c839c33742309ec41e620bf837495ec9926c56e`</sub>

### ❌ hd256 seqused 2cta constraint

工况: FA4(CuTeDSL,vllm fork pin 2c839c33)在 Blackwell(SM100/SM101)上 head_dim=256 走的是 dedicated 2-CTA forward kernel;bf16/fp8,varlen+paged-KV。
方法: 核实该专用 kernel 路径是否接受 seqused_q/seqused_k。
预期: 判断 hd256+paged-KV 能否带 seqused 在 FA4 上跑(vLLM decode 的典型需求)。
实测: 不支持。interface.py:978-979 硬断言 assert seqused_q is None and seqused_k is None,报错 'SM100 forward with head_dim=256 does not support seqused_q/seqused_k';同路径同时禁用 softcap / block sparsity / learnable_sink。这解释了 vllm fa_utils.py 为何在 SM100 上把 head_size>128 且 ≠192 直接降级到 FA2。
归因与结论: 负向/约束。根因: hd256 因 TMEM 容量必须走 2-CTA 专用 kernel,而该 kernel 未实现 seqused;若强行不带 seqused,paged-KV 的未初始化 padding 会在 P@V 里产生 0*NaN(对应 memory fa4-hd256-seqused-padding-nan-bug),故上游选择直接断言禁用而非静默出错。可操作规则: 在 Blackwell 上做 hd256 paged-attention 且需要 seqused/变长有效长度时,不能指望官方 FA4——要么落到 FA2、要么自研 hd256 kernel 并显式处理 padding;'FA4 支持 varlen+seqused'这一整体结论在 hd256 2CTA 路径上不成立。

<sub>`tmem` `cta_cooperation` `x-tmem_capacity` `numerical_instability`  session `89c40dec` · commit `2c839c33742309ec41e620bf837495ec9926c56e`</sub>

