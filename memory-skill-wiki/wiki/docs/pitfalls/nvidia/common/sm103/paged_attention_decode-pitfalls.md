# Paged Attention Decode on sm103 — Pitfalls

Traps hit while optimizing `paged_attention_decode` — flashinfer trtllm-gen (prebuilt cubin) baseline, bf16, 31 production shapes; failures concentrated on block256 / klen1024 大 shape on B300 / Blackwell / sm103.
Companion to:

- Optimization journey: [journey](../../../../ref-docs/nvidia/common/sm103/paged_attention_decode-optimization.md)

---

## 1. trtllm-gen cubin-version-mismatch

**Trap**: 在本机复现 trtllm-gen 基线,尝试 flashinfer 0.6.9 与源码 JIT 版（预期 trtllm_fixed.py 直接跑出记录基线）

**Result**: 0.6.9: 15/31 shape 报 loadKernel 失败(cubin 缺该配置),能跑的也 ~55µs;FI_FORCE_JIT=1 无效(trtllm-gen 是 NVIDIA 预编译 cubin,不能源码 JIT);升级 0.6.12 才 31/31 全过

**Why**: trtllm-gen kernel 以预编译 cubin 分发,cubin 与具体 flashinfer 构建/shape 配置强绑定;trtllm_fixed.py 针对北园那套 flashinfer 编写,源码版(0cb2bc9)又与 0.6.9 后端 FFI 签名不匹配

**Lesson**: 负向(环境/工具链坑,非算法问题)。trtllm-gen 是 NVIDIA 预编译 cubin,无法从源码 JIT,cubin 与 flashinfer 构建版本、shape 配置强绑定。教训: 复现 trtllm-gen 基线必须锁定与之匹配的 flashinfer 构建;loadKernel 失败=cubin 缺该 shape 配置,应先升级/换构建而不是改代码;跨环境搬运 trtllm_fixed.py 这类薄封装时,FFI 签名与 cubin 版本要一并对齐。

<sub>`x-toolchain`  session `6270b3b2`</sub>

## 2. trtllm-gen seqused-host-sync

**Trap**: 诊断 flashinfer trtllm-gen 基线为何延迟恒定、远高于记录值（预期 复现北园记录的 trtllm-gen ~19.3µs mean 基线）

**Result**: 实测各 shape 几乎恒定 ~55-68µs,不随 KV 长度变化;flashinfer 0.6.12 修好 loadKernel(31/31 全过)后仍 ~62µs,而非 ~19µs

**Why**: trtllm_fixed.flash_decode 里 seqused_k.max().item() 每次调用都触发 GPU→CPU 同步 → 主机/launch 受限;延迟恒定、不随 workload 变化正是典型 launch-bound 信号,而非 GPU kernel 真实时间

**Lesson**: 负向。恒定且与 workload 无关的延迟是 host/launch-bound 的判据;根因是 trtllm_fixed.flash_decode 中 seqused_k.max().item() 每次调用做 GPU→CPU 同步。教训: 评估 decode 基线时先看'延迟是否随 seqlen 变化',不变即怀疑 host 同步/launch 开销;.item()/.max().item() 这类隐式同步必须在计时路径外消除,否则测到的是主机延迟而非 kernel 性能。

<sub>`launch_optimization` `x-host-sync` `launch_overhead` `latency_bound`  session `6270b3b2`</sub>

## 3. late fa4 hd256 prefill only for decode

**Trap**: 尝试直接复用 FA4 现成 hd256 2CTA kernel 跑 beiyuan 31 shape decode(flash_attn_varlen_func + page_table),试 num_splits、pack_gqa、re-page 到 page_size=128（预期 FA4 hd256 是现成的 TMA-based D=256 paged 核,或可直接给出 <30µs decode）

**Result**: rel=0.0021 PASS,但 klen=1024 ~400µs vs a4 104µs(~4× 慢);num_splits>1 报错(不支持 SplitKV);pack_gqa 无差别(398 vs 404µs);seqused_k 不支持但 decode 各 batch KV 均匀=klen 可丢

**Why**: FA4 hd256 是 prefill 专用核,无 SplitKV/decode 优化,对 q_len=4 的 decode 极低效;故不存在可直接复用的 D=256 decode-优化核,a4 已胜过它

**Lesson**: 负向。FA4 hd256 是 prefill 专用核,无 SplitKV、无 decode(q_len 小)优化,对 q_len=4 的 decode 极低效;因此不存在可直接复用的 D=256 decode-优化核,a4 已经胜过它。教训: 复用现成 attention 核前先确认它是 prefill 还是 decode 形态——prefill 核缺 SplitKV,对 q_len 极小的 decode 会因 grid 欠占用/无 flash-decoding 而慢数倍,正确不等于快;<30µs 只能从零建 trtllm 级 decode 核。

<sub>`split_kv` `x-kernel-reuse` `grid_underutilization` `latency_bound`  session `3e24042c`</sub>

