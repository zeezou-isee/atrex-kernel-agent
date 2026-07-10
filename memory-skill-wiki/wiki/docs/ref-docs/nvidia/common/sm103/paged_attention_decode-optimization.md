# Paged Attention Decode on sm103 — Optimization Journey (ref-docs)

**Last Updated**: 2026-07-03  ·  3 recorded attempt(s)

Hardware: B300 / Blackwell / sm103 · DSL: flashinfer trtllm-gen (prebuilt cubin) baseline · dtype: bf16 · shapes: 31 production shapes; failures concentrated on block256 / klen1024 大 shape

---

## Attempts in detail

### ❌ trtllm-gen cubin-version-mismatch

工况: 在 B300/sm103 本机复现 flashinfer trtllm-gen paged decode 基线,31 个 production shape,大 shape 为 block256/klen1024。
方法: 先用已装的 flashinfer 0.6.9 直接跑 trtllm_fixed.py,再试源码 JIT 版与 FI_FORCE_JIT=1。
预期: 薄封装 trtllm_fixed.py 应直接产出基线数值。
实测: 0.6.9 下 15/31 大 shape 报 loadKernel 失败(cubin 缺配置),能跑的小 shape ~55µs;FI_FORCE_JIT=1 无效;必须升级到 0.6.12 才 31/31 全过。
归因与结论: 负向(环境/工具链坑,非算法问题)。trtllm-gen 是 NVIDIA 预编译 cubin,无法从源码 JIT,cubin 与 flashinfer 构建版本、shape 配置强绑定。教训: 复现 trtllm-gen 基线必须锁定与之匹配的 flashinfer 构建;loadKernel 失败=cubin 缺该 shape 配置,应先升级/换构建而不是改代码;跨环境搬运 trtllm_fixed.py 这类薄封装时,FFI 签名与 cubin 版本要一并对齐。

<sub>`x-toolchain`  session `6270b3b2`</sub>

### ❌ trtllm-gen seqused-host-sync

工况: B300/sm103 上以 flashinfer trtllm-gen(预编译 cubin)作为 paged decode 基线,31 个 production shape,hd256/GQA/q_per_seq=4/block 64|256。
方法: 对基线做延迟诊断——逐 shape 测均值并观察延迟随 KV 长度的变化趋势。
预期: 应复现记录中的 trtllm-gen ~19.3µs mean。
实测: 各 shape 延迟几乎恒定在 ~55-68µs,完全不随 KV 大小增长;换 flashinfer 0.6.12 修好 loadKernel(31/31 通过)后仍 ~62µs,与 19µs 相差 3× 以上。
归因与结论: 负向。恒定且与 workload 无关的延迟是 host/launch-bound 的判据;根因是 trtllm_fixed.flash_decode 中 seqused_k.max().item() 每次调用做 GPU→CPU 同步。教训: 评估 decode 基线时先看'延迟是否随 seqlen 变化',不变即怀疑 host 同步/launch 开销;.item()/.max().item() 这类隐式同步必须在计时路径外消除,否则测到的是主机延迟而非 kernel 性能。

<sub>`launch_optimization` `x-host-sync` `launch_overhead` `latency_bound`  session `6270b3b2`</sub>

### ❌ late fa4 hd256 prefill only for decode

工况: B300/sm103,评估复用 FA4 的 hd256 2CTA forward kernel 跑北园 31 shape paged decode(hd256/GQA/q_per_seq=4/klen 256|1024/page 64|256)。
方法: 用 flash_attn_varlen_func(q, k=key_cache, v=value_cache, cu_seqlens_q, page_table=block_table, causal=True)映射北园 paged_attention 签名;因 FA4 hd256 需 page_size=128,把 KV cache re-page 到 128;seqused_k 不支持但 decode 各 batch KV 长度均匀=klen 可丢弃、用 max_seqlen_k=klen;并试 num_splits(flash-decoding)与 pack_gqa。
预期: FA4 hd256 是现成的 TMA-based D=256 paged 核,或能直接给出 <30µs decode,省去从零重写。
实测: 正确性 rel=0.0021 PASS,但性能差——klen=1024 ~400µs vs a4 的 104µs(~4× 慢);num_splits>1 报错(FA4 hd256 不支持 SplitKV);pack_gqa 无差别(398 vs 404µs)。
归因与结论: 负向。FA4 hd256 是 prefill 专用核,无 SplitKV、无 decode(q_len 小)优化,对 q_len=4 的 decode 极低效;因此不存在可直接复用的 D=256 decode-优化核,a4 已经胜过它。教训: 复用现成 attention 核前先确认它是 prefill 还是 decode 形态——prefill 核缺 SplitKV,对 q_len 极小的 decode 会因 grid 欠占用/无 flash-decoding 而慢数倍,正确不等于快;<30µs 只能从零建 trtllm 级 decode 核。

<sub>`split_kv` `x-kernel-reuse` `grid_underutilization` `latency_bound`  session `3e24042c`</sub>

