# sm103 flash attention prefill — Optimization Highlights (kernel-opt)

> Optimization highlights (one entry per proven technique). Full version-by-version journey + pitfalls: see Further Reading.

**Further Reading**: [journey](../../../../ref-docs/nvidia/cutedsl/sm103/flash_attention_prefill-optimization.md) · [pitfalls](../../../../pitfalls/nvidia/cutedsl/sm103/flash_attention_prefill-pitfalls.md)

---

## hd256 1cta fix

**Trigger**: numerical_instability, grid_underutilization — real vLLM serving (prefill + decode) / hd256, GQA (qwen3.7-max/3.5-plus), paged KV + varlen + seqused_k

**Technique**: hd256 bf16 从 2CTA 专用内核改走 1-CTA:env FA4_HD256_1CTA=1 时置 use_dedicated_hd256_kernel=False, use_2cta_instrs=False, tile_n=128, q_stage=1,经通用 FlashAttentionForwardSm100 以 CtaGroup.ONE 跑 hd256(默认关,不改 vLLM 行为)。bf16 强制 q_stage=1(q_stage=2 光 Q+O 就要 256KB SMEM > 224KB 上限)

**Expected**: 1-CTA 每个 CTA 在自己 TMEM 里算完整 tile,无 peer-CTA、无跨 CTA accumulator、无 DSMEM 耦合 → 对并发 TMEM 用户免疫 → 修掉 2CTA 的乱码

**Effect**: test_seqused_paged.py 14/14 PASS(PAD_FILL=garbage/zero,max|diff|~1e-2 bf16;PAD_FILL=nan 仅 1/14 因通用内核缺末尾 partial-tile flush,但 vLLM KV cache 是 torch.zeros → 无关);真实 pai-vllm qwen3.7-max TP4 端到端 3 prompt 全部输出连贯结构化 token,乱码消失。serving 路径 1-CTA vs 2-CTA:decode 快 4.2-6.4x,prefill 快 1.5-2.3x,仅大 dense 单序列 prefill 2-CTA 快 1.1-1.3x

**Why it works**: 正向,是乱码的正解:CtaGroup.ONE 消除了 2SM MMA 的 peer-CTA 半区,不再依赖跨 CTA TMEM/DSMEM 相干 → 并发 TMEM 用户无法腐蚀。serving 下更快因 decode 时 M=1 用 2CTA 是浪费,且 2CTA varlen 走慢的 SingleTileVarlenScheduler;2CTA 只在大 dense prefill 的 throughput/ M-split 上占优

<sub>`cta_cooperation` `tmem` `tiling` `numerical_instability` `grid_underutilization`  session `4b8ff1d7` · commit `2b1511f`</sub>

## hd256 1cta tile n96

**Trigger**: pipeline_stall, barrier_sync, smem_lds_capacity — serving prefill (1-CTA path) / hd256, GQA 32Q/2KV (biz [B=1,S] S in {7680,8064,8320,8576}), paged+varlen

**Technique**: 在 1-CTA hd256 路径把 tile_n 从 128 降到 96(env FA4_TILE_N,后设为默认):tile_n<128≠page_size 会掉出 TMA、改走非-TMA paged gather(cp.async),但把每级 KV-SMEM 减半 → 容得下 kv_stage=2 预取 + 腾出 TMEM 余量

**Expected**: kv_stage=2 增加 KV 预取重叠,缓解 MMA 每个 KV 块等 softmax 产 P 的 barrier stall

**Effect**: 4 biz shape:latency 0.848/0.928/1.014/1.032ms vs baseline(tile_n=128)0.99/1.08/1.15/1.21 → 快 12-18%,1119-1167TF;gap 对 trtllm 从 1.52->1.29x(S7680)、1.32->1.18x(S8576)。ncu@7680:Compute SOL 46.8->59.6%,tensor active 44.5->53.9%,barrier stall 7.31->4.08,long_scoreboard 8.93->6.19,occ 15.5->19.9%(>trtllm 18.6),dyn smem 198->231KB。test_seqused_paged 14/14 PASS,数值与 tile_n=128 一致。真实 TP4 端到端 A/B:tile_n=96=12.03us/call vs 128=13.04us/call(~8% 更快),输出 token 完全一致

**Why it works**: 正向:tile_n=96 是仍能给出 kv_stage=2 的最大 tile(112->kv_stage=1 回退,64->kv_stage=3 迭代太多 1052-1066TF);kv_stage=2 的预取重叠把 barrier stall 砍近半、把 tensor-core 空转从 55% 降到 46%。代价是掉 TMA 走 cp.async gather(L1 hit 降),但净收益为正

<sub>`tiling` `multi_buffering` `overlap_pipelining` `pipeline_stall` `barrier_sync` `smem_lds_capacity`  session `4b8ff1d7` · commit `2b1511f`</sub>

## v1 exp2 freq tuning

**Trigger**: compute_bound, pipeline_stall, occupancy_limited — prefill self-attention (dense) / hd256, GQA 32Q/2KV, tile 128x128x256, 2CTA, S in {7680,8064,8320,8576}

**Technique**: V1 parameter_tuning: 把 hd256 causal 的软件 exp2 interleave 频率 ex2_emu_freq 从 14 调到 20(_TUNING_CONFIG),让 softmax warp 的 6 阶 Horner 多项式 exp2 相对 MMA correction 交织得更疏,减少 FMA 单元争用

**Expected**: 2-8% 小幅提升,通过更好的 MMA/exp2 交织平衡;内核根本被 18.75% occupancy(256 regs/softmax warp + 384KB SMEM 跨 2CTA)限死,无法大改

**Effect**: V0->V1: S7680 668->666us(-0.3%), S8064 829->808us(-2.5%), S8320 880->873us(-0.8%), S8576 946->868us(-8.2%);均值 TFLOPS 1324->1364(+3%);rel_l2 不变(0/regression)。ncu:SM Throughput 74.98%,No-Eligible 61.36%,首要 stall 是 L1TEX scoreboard(占 7.8cyc 平均 warp stall 的 39.1%)

**Why it works**: 正向但很小:exp2 用 FMA,和 correction warp 的 FMA 抢单元;降交织频率给 MMA 流水更连续的执行窗口。但 SM throughput 已 75%、只 57-64% 峰值,因 12 warp 里仅 1 个做 MMA、软件 exp2 的 FMA 不计入 FLOP 预算、且 39% 的 L1TEX scoreboard stall 是生产者(TMA)/消费者(MMA)气泡——这些参数调不动

<sub>`fast_math` `mma_scheduling` `compute_bound` `pipeline_stall` `occupancy_limited`  session `4b8ff1d7`</sub>

## v2 sm103 hw exp2

**Trigger**: compute_bound, pipeline_stall — prefill self-attention (dense) / hd256, GQA 32Q/2KV, 2CTA, S in {7680,8064,8320,8576}

**Technique**: V2 arch_specific: 给 hd256 2CTA 内核加 SM103(B300)检测(BaseDSL._get_dsl().get_arch_enum(), is_family_of(sm_103f)),在 SM103 上关闭软件 exp2 模拟(ex2_emu_freq=0)改用硬件 SFU exp2。FA4 的非-hd256 内核早有此优化(flash_fwd_sm100.py:205),但 hd256 专用内核硬编码 is_sm103=False 从未应用

**Expected**: 对 softmax 计算占比大的 shape(S>=8320)10-20% 提升;SM103 的 SFU 比 SM100 快,14 系数 Horner 软件模拟成了纯开销,走硬件 SFU 可把 FMA 单元让给真正有用的 correction 工作

**Effect**: A/B 同 seed=42 warmup=50 rep=200: S8320 926->782us(+15.5%,1225->1450TF), S8576 961->824us(+14.3%,1254->1463TF), S7680 669us 不变(小 shape exp2 占比可忽略), S8064 -1.3%(噪声);rel_l2 ~0.002 不变,硬件 exp2 精度足够

**Why it works**: 正向:SM103 SFU 吞吐更高时,软件 exp2 emulation 是纯 FMA 开销;设 ex2_emu_freq=0 走硬件 exp2 → softmax warp 更快完成 → correction/MMA 前的流水气泡变小。收益集中在 softmax 占比大的长 shape,短 shape 无感

<sub>`fast_math` `mma_scheduling` `compute_bound` `pipeline_stall`  session `4b8ff1d7` · commit `2b1511f`</sub>

## vllm fa4 build

**Trigger**: x-build-integration — 编译 vllm + fa4(vllm_flash_attn)进 conda env xingze(py3.12, torch 2.11.0+cu130, nvcc 13.2) / n/a

**Technique**: 源码编译 vllm 与 fa4 进 conda env xingze:vllm 先跑 use_existing_torch.py 剥 torch 钉再 pip install -e . --no-build-isolation;fa4 用 pip install -e . --no-build-isolation --no-deps

**Expected**: 两个包 editable 装进现有 env,不动已装的 torch 2.11 / CUDA 栈

**Effect**: 踩到 4 个坑:(1) vllm 需 setuptools>=77(env 是 70.2.0,PEP-639 的 license='Apache-2.0' 报错)→ 升到 80.10.2;(2) fa4 不加 --no-deps 会去装 metadata 钉的 torch==2.4.0(797MB)覆盖 torch 2.11 → 必须 --no-deps;(3) vllm ~76 target ~50min、fa4 251 target ~3h(CPU quota 限 ~1 core,MAX_JOBS 无用);(4) 编好后 test_fa4_smoke 仍失败——bundled flash_attn 的 __init__.py 硬 import legacy flash_attn_2_cuda(本仓库不构建、env 也没有)→ 用 try/except 包住该 import 才能用 flash_attn.cute

**Why it works**: torch 覆盖坑源于 FA4 metadata 钉死旧 torch,--no-deps 是唯一防线;setuptools 坑源于新 pyproject 的 PEP-639 SPDX license 字段;smoke test 失败是父包 __init__ 贪婪 import 一个只有 PAI wheel 才有的 legacy .so,与本次构建无关

<sub>`x-build-integration` `x-build-integration`  session `4b8ff1d7`</sub>

## vllm hd256 seqused integration

**Trigger**: x-build-integration — vLLM serving (prefill + decode) integration / hd256, GQA, paged KV (page_size==128, block_table row-major), varlen + seqused_k

**Technique**: 把 FA4 hd256 接进 vLLM 的 paged/varlen 注意力路径:vLLM 的 FA4 wrapper(fa_utils.py:145)总是给内核传 seqused_k,但 FA4 baseline hd256 2CTA 内核硬 assert 拒绝 seqused_q/seqused_k(sm100_hd256_2cta_fmha_forward.py:189, interface.py:978)

**Expected**: 让 hd256 能在 vLLM 的变长 paged 路径跑起来

**Effect**: warmup 直接崩:'SM100 forward with head_dim=256 does not support seqused_q/seqused_k'。两条路:(1) prefill-only 权宜——把 seqused_k 从 wrapper 签名里去掉使 has_seqused_k=False(该标志由 inspect.signature 探测),vLLM 转而传 cu_seqlens_k(hd256 内核支持),max_tokens=1 的 prefill 用 cu_seqlens_k 足以描述 KV → 跑通;(2) 根治——fa4_bf16_varlen 分支(commit 2b1511f)给 hd256 内核补齐 seqused_q/k:kernel 参数 + 四处 warp 段 per-batch 覆写 seqlen_k=seqused_k[batch_coord],经 get_trip_start_count_via_block_info 真正截断 KV 循环。但 backward 仍 assert 拒绝(仅前向支持)

**Why it works**: 根因:vLLM 假设 'FA 内核用 seqused_k 界定 KV walk',这对 FA4 hd≤128 内核成立,但 hd256 专用 2CTA 内核从未实现 → decode(仅靠 seqused_k 编码变长)完全无法驱动。hd256 paged 还要求 page_size==128(TMA-only)且 block_table row-major 连续

<sub>`paged_gather` `x-varlen-support` `x-build-integration`  session `4b8ff1d7` · commit `2b1511f`</sub>

