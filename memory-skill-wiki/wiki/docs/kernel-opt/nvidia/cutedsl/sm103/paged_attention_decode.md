# sm103 paged attention decode — Optimization Highlights (kernel-opt)

> Optimization highlights (one entry per proven technique). Full version-by-version journey + pitfalls: see Further Reading.

**Further Reading**: [journey](../../../../ref-docs/nvidia/cutedsl/sm103/paged_attention_decode-optimization.md) · [pitfalls](../../../../pitfalls/nvidia/cutedsl/sm103/paged_attention_decode-pitfalls.md)

---

## late combine kernel ndchunk occupancy

**Trigger**: occupancy_limited, grid_underutilization — decode combine (flash-decoding reduction) / hd256 (D=256), GQA, split-KV num_splits, combine 归约

**Technique**: 把 flash-decoding combine 写成独立 kernel(grid=(B,nkv,N_DCHUNK),两遍归约:phase1 算 global-max + per-split rescale;phase2 按 D-chunk 加权累加),扫 N_DCHUNK 提占用率

**Expected**: combine 是小-work 归约,靠更多 CTA 填满 SM 降低 launch/occupancy 受限的耗时

**Effect**: combine 26.65µs(N_DCHUNK=4,64 CTA)→ 18.30(=8)→ 12.43(=16)→ 10.34µs(=32,512 CTA);≥32 后 ~10.3µs 平台。full pipeline 53.31→43.15µs,31/31 PASS rel≈0.0021

**Why it works**: 初始 64 CTA 只有极少 work、SM 大量闲置(occupancy-bound);增大 N_DCHUNK 把 D 维拆更多 CTA 填满 148 SM。坑:N_DCHUNK 必须整除 D=256,否则漏列/越界(48→DC=5 只覆盖 240 列,是静默 bug),取 N_DCHUNK=32(DC=8 精确)

<sub>`occupancy_tuning` `split_k` `online_softmax` `occupancy_limited` `grid_underutilization`  session `3e24042c` · commit `2b76d06`</sub>

## late kstages2 desqn smem optin

**Trigger**: smem_lds_capacity, pipeline_stall, latency_bound — decode stage (warp-spec a3) / shape0 klen=256(4 tile), hd256, M_TILE=128, 16 CTA

**Technique**: a3 去掉 sQn(64KB Q staging):Q 改每个 compute 线程从 gmem scalar-gather 自己那行→寄存器→relayout 进 swizzled sQ;腾出空间开 KSTAGES=2 双缓冲

**Expected**: 去 sQn 省 64KB 并顺带消 Q-pack race;KSTAGES=2 让 load(t+1) 与 compute(t) 重叠

**Effect**: shape0 decode-path 57.7µs → 30.6µs(1.9×,首次超过 stage4 43µs),rel=0.002102 PASS

**Why it works**: 双缓冲 load/compute 重叠 + Q 改寄存器直取(省 scalar pack)收益显著,即使还只有 16 CTA。关键坑:launch 的 smem 必须设到 SM100 max optin 232448 字节(实际需 ~229376);此前设 228000 < 实际需求导致静默 OOB(编译不报,运行时 CUDA illegal memory access)

<sub>`multi_buffering` `async_copy` `smem_lds_capacity` `pipeline_stall` `latency_bound`  session `3e24042c` · commit `f68fb22`</sub>

## late splitkv occupancy beats overlap

**Trigger**: grid_underutilization, occupancy_limited, latency_bound — decode stage (a4 = warp-spec decode + splitKV + combine) / shape0 klen=256 B=8, hd256, GQA; klen 256/1024

**Technique**: 把 a3 warp-spec decode 塞进 stage4 的 splitKV 骨架(grid=(B,nkv,num_splits),写 partial + 复用 combine),实测 NS=4(1 tile/split,64 CTA)/ NS=2(2 tile,32 CTA,带跨-tile 重叠)/ NS=1(4 tile,16 CTA)

**Expected**: 对比高占用率(splitKV 多 split)与跨-tile 重叠(KSTAGES=2)哪个对小 klen 更划算,用数据定

**Effect**: shape0 NS=4=18.56µs、NS=2=22.5µs、NS=1=32µs;分解 decode 15.3 + combine 10.5 = full ~19µs。轨迹 stage4 43→a3(KSTAGES2)30.6→a4(splitKV)18.56µs。klen 256/1024 都 PASS

**Why it works**: 小 klen 下提占用率(splitKV 到 64 CTA)决定性胜过跨-tile 重叠——KSTAGES=2 重叠需每 CTA ≥2 tile,与高占用率二选一;这正是 trtllm 的路子(1 tile/split + 靠 warp 藏单-tile 延迟)

<sub>`split_kv` `grid_underutilization` `occupancy_limited` `latency_bound`  session `3e24042c` · commit `f68fb22`</sub>

## late v10 readback dst shape bug

**Trigger**: numerical_instability — decode stage / hd256, GQA, q_per_seq=4, block_n=64, block_size 64/256, split-KV

**Technique**: 定位并修复上一轮遗留的 Stage4 split-KV 'PV→O=0' bug;用独立读路径隔离 PV / readback / 绑定三段

**Expected**: 让 tcgen05 split-KV decode 端到端正确(rel_l2 ≤ 0.02)

**Effect**: rel_l2 1.000000 FAIL → 0.002025 PASS,随后全 31/31 shape PASS(rel≈0.002);stage3a 对照仍 21µs PASS,证明环境无变化

**Why it works**: 真根因是 O 的 readback 寄存器张量按源(TMEM)分片 shape 建,而非目的(gmem)分片 shape → Ld16x64 epi-copy 静默读回 0;上一轮误判为多-CTA/PV/TMEM。破局关键:先用另一条已验证的 Ld32x32 逐行读路径直接读 TMEM_O(sumsq=32.14 非零)证明 PV 一直正确,从而把 bug 锁死在 readback 而非 PV/绑定

<sub>`tmem` `numerics_fix` `in_kernel_fusion` `numerical_instability`  session `3e24042c` · commit `2b76d06`</sub>

## late warpspec decode a0 a3 build

**Trigger**: pipeline_stall, latency_bound, numerical_instability — decode stage / hd256 (D=256), GQA, q_per_seq=4, block_n=64, klen 256(4 tile)/1024(16 tile)

**Technique**: 从零分步搭 warp-specialized 分页 decode 核(a0 骨架→a1 mainloop→a2 D=256→a3 真实分页):专用 LOAD warp(cp.async 生产者)+ COMPUTE group(warp0-3 交织 QK→online-softmax→PV→O)+ 手写 mbarrier 2-stage 双缓冲;D=256 用 per-D-chunk Ld32x32/St32x32 做 TMEM-O rescale;接 GQA-pack+paged gather+V转置+causal-tail

**Expected**: 建立 16-warp 重写的地基:先证明 warp-specialize load↔mma 双缓冲流水与 tcgen05 共存、且多 tile online-softmax 数值正确

**Effect**: a0 last-tile rel=0.0 PASS(且区别于其他 tile,证明 staging 正确);a1 4-tile rel=0.0014;a2 D=256 rel=0.0011;a3 对 reference.py klen=256 rel=0.002102、klen=1024(16 tile)rel=0.002105,均 PASS(16-tile constexpr 约几分钟编译不爆 IR)

**Why it works**: warp-specialize 让 load(t+1) 与 compute(t) 重叠是藏延迟的雏形;per-D-chunk Ld32x32(1 线程/行)让 corr 天然按行落在每线程手里、绕开 FA4 correction warp 的 sScale smem 广播

<sub>`warp_specialization` `multi_buffering` `async_copy` `paged_gather` `tmem` `online_softmax` `pipeline_stall` `latency_bound` `numerical_instability`  session `3e24042c` · commit `f68fb22`</sub>

## pv v orientation rule

**Trigger**: x-operand_layout, uncoalesced_access — decode stage (PV operand orientation) / single-tile PV probe (D=64), then hd256

**Technique**: 确定性最小诊断: 令 V[n,d]=d+1(V[0,:] 随 d 变、V[:,0] 恒定,易区分)、P 只选 key 0,直接读 O 判定 tcgen05 PV MMA 把 sV 读成 V 还是 V^T;并对 b_major=K 与 MN 各测一遍

**Expected**: 把 PV 的 V-operand 朝向彻底定死,好让 paged gather 正确写 sV(这是卡住 Stage 3b 的谜题)

**Effect**: 定论(已验证): tcgen05 PV MMA 计算 P @ (sV)^T,与 b_major 无关(K 和 MN 都给 P@V^T)。要得 attention 的 P@V,sV 必须物理存 V^T。喂 V 自然 -> O=[1,1,1,...]=P@V^T(错);喂 V^T 或 stride 交换的转置视图 sVn_T 作源 -> O=[1,2,3,...]=P@V(对,rel 0.0018)

**Why it works**: autovec_copy 用相同 TV 布局是按物理序拷贝(等于恒等,不做逻辑转置);真正转置需要 stride 交换的源视图(sVn_T)、ldmatrix.trans 或 TMA —— cp.async 无法转置。朝向由 sV 的物理布局决定,不由 b_major,这解释了 b_major 看似无效的悖论

<sub>`swizzle` `tcgen05_umma` `x-operand_layout` `uncoalesced_access`  session `6270b3b2`</sub>

## softmax threadlocal reduction

**Trigger**: latency_bound — decode stage (online softmax) / M=128 N=64, multi-tile klen=128/256 (D=64 isolation)

**Technique**: 在 TMEM-read 的 S fragment 上做 online softmax;先查 Ld32x32bOp 在 (128,64) S tile 上给每线程的 fragment 布局,再决定行归约是跨线程 shuffle 还是线程内

**Expected**: 搞清 row max/sum 归约该怎么做,并跑通多-tile online softmax(running max/sum + correction + 寄存器 O 累加)

**Effect**: 发现: Ld32x32bOp 读 (128,64) S 时,128 个线程每个恰好持有一整行的 N=64 个值(rS=64 elem;128x64=M*N)-> softmax 行归约 + per-thread O correction 完全线程内、无需跨线程 shuffle。多-tile online softmax 2-tile 与 4-tile 均 rel_l2约0.001 PASS

**Why it works**: N=64 的 tile 宽度让每行恰好落一个线程,running-max/running-sum/correction 都是 per-thread 标量 —— 比 FA4 SoftmaxSm100 的 warp-specialized 跨 lane 归约简单得多;correction O=O*corr+O_t 在 D=64 可在寄存器做(D=256 则不行,见 hd256 预算记录)

<sub>`online_softmax` `warp_reduction` `tmem` `latency_bound`  session `6270b3b2`</sub>

## v10 inkernel pv fusion cracked

**Trigger**: latency_bound, register_pressure — decode stage (in-kernel FMHA fusion) / single-tile M=128 N=64 D=64 (isolated), base 9a14c6d

**Technique**: 攻克北园 V10 从未跑通的 in-kernel tcgen05 PV 融合。关键: (1) S 与 O 放 DISJOINT TMEM 列区(S[0,64)/P bf16[64,96)/O[128,384)),不共用一块;(2) M-tile 对齐 128(干净 fragment)via make_trivial_tiled_mma —— M=64 会得复合 (16,4) fragment,Ld32x32/St32x32 atom 不适配;(3) P 经 row-bridge 写回 TMEM: Ld32x32bOp 读 S、寄存器 exp2、把 bf16 P 打包成 fp32-word(rP_bf16 与 rP_f32 共享同一寄存器)用 St32x32bOp(Float32) 存;P 作为 PV A-operand 用 fp32 的 S iterator + bf16 A-layout 读回(recast_ptr(bf16) 会翻倍 stride,错);(4) PV MMA 走 TS-mode,a_major=K

**Expected**: 解掉 V10 文档记录的两个 bug(TMEM 复用/重置导致 rel_l2=1.5、PV 坐标映射错),让单-tile in-kernel FMHA 正确

**Effect**: 单-tile in-kernel FMHA(QK->exp2 softmax->PV->O readback)rel_l2=0.0018 PASS(v1 unnorm,再加真 softmax v2 仍 0.0018);do_bench 11.2us 但 LAUNCH-BOUND(单 CTA tiny tile,约等 QK-only 11.4us,GPU 计算 <1us)—— 是正确性突破,不是性能数

**Why it works**: V10 的 rel_l2=1.5 来自把 S 和 O pack 进同一块 512 列 TMEM 并指望 ACCUMULATE=False 重置(它只影响下一次 MMA 的首个 k-block,不清跨-tile 的区域)—— S/O 分列 + O 只清零一次即解。atom 不匹配根因是 M=64 的非-128 复合 fragment;M=128 干净 fragment 让 FA4 的 Ld32x32/St32x32 适配。recast_ptr(bf16) 造 P 的 A-operand 会把 M-stride 翻倍(131072 vs 65536),必须用 fp32 S iterator + bf16 A-layout

<sub>`tcgen05_umma` `tmem` `in_kernel_fusion` `latency_bound` `register_pressure`  session `6270b3b2`</sub>

## v1 vectorized gather

**Trigger**: memory_bound, latency_bound, uncoalesced_access — decode stage / hd256, GQA, q_per_seq=4, block_n=64, block_size {64,256}, 31 shapes

**Technique**: 把 V0 的逐元素 scalar paged gather 换成向量化 128-bit 合并的 KV gather(每次 32 线程 x 8 bf16/行,单物理块 (N,D) block-slice)

**Expected**: 消除 scalar gather 的 uncoalesced 访存(V0 主导的 long_scoreboard 9.73),把 memory_bound 打下去

**Effect**: mean ~196us(shape6 222.7us),相对 V0 ~2400us 约 4-10.8x;big-shape 达 543 GB/s;long_scoreboard 从 9.73 大幅塌缩;rel_err ~0.002 不变,31/31

**Why it works**: 128-bit 合并 load 把 LD 数量降约 8x,直接去掉 V0 的 uncoalesced-access 主瓶颈;但暴露出新的 latency/小-grid 限制(waves/SM 0.22、~6.3% occupancy、235 regs、load 与 MMA 无重叠)

<sub>`vectorized_load` `paged_gather` `memory_bound` `latency_bound` `uncoalesced_access`  session `6270b3b2` · commit `14ea770`</sub>

## v2v3 launch overhead

**Trigger**: launch_overhead, latency_bound — decode stage / hd256, GQA, q_per_seq=4, block_size {64,256}, 31 shapes

**Technique**: V2: 去掉 paged_attention() wrapper 里每次调用的 cu_seqlens_q.tolist() GPU->CPU 同步,改成 sync-free 的 q_per_seq=T//B(31 shape 均为均匀连续前缀和);由此可 cudagraph capture。V3: 加 _LAUNCH_CACHE(按 data_ptr 键)让热路径跳过 ~112us 的 from_dlpack x7 + dim_order + mark_compact + empty_like 重建

**Expected**: 填掉 ~141us/call 的 host gap(ncu GPU 81.8us vs do_bench wall 223us),并让函数可被 CUDA graph 捕获

**Effect**: V2 eager 196->100.7us(1.95x)、cudagraph 首次可用 mean 67.0us;V3 eager 66.6us == cudagraph 67.1us(host 开销消除);rel_err 0.00213,31/31。commit 029ff9d(V2)/17b7f90(V3)

**Why it works**: .tolist() 每次调用强制 device 同步(且 .item()/.tolist() 会让 cudagraph capture 报错);from_dlpack x7 等每次重建全部 cute tensor ~112us —— 按 data_ptr 缓存已准备好的 launch 后 eager wall 就等于 GPU 时间,与 trtllm 的差距变成纯 GPU-kernel 问题

<sub>`launch_optimization` `launch_overhead` `latency_bound`  session `6270b3b2` · commit `17b7f90`</sub>

## v4 cpasync pipeline

**Trigger**: latency_bound, smem_lds_capacity — decode stage / hd256, GQA, q_per_seq=4, block_n=64, block_size {64,256}, 31 shapes

**Technique**: 把同步 128-bit KV gather 改成 cp.async 2-stage(双缓冲)软件流水: gather atom 换 CopyG2SOp(cache_mode=GLOBAL) num_bits=128;sK/sV 变成 N_STAGES=2 的 staged swizzled 布局(追加 stage 模,运行时 sK[None,None,stage] 选缓冲);prologue 载 tile0 + cp_async_commit_group,循环内预取 n+1、cp_async_wait_group(1) 让预取在飞、drain 当前、barrier

**Expected**: collapse long_scoreboard(3.36,主导 warp stall):让下一 KV tile 的 HBM 读在当前 MMA 期间在飞,把 MLP 从 1 提到 2,把有效带宽抬离 543 GB/s / 7.5% SOL 的地板,尤其 page256 的 16-tile shape

**Effect**: eager 66.6->47.1us(1.41x),cudagraph 67.1->47.4us(1.42x);shape6 89.6->59.8(1.50x),有效带宽 543->1192 GB/s(14.9% SOL);shapes 9/30 ->2030 GB/s(25-28% SOL);tflops 71.8;无 shape 回退;rel 0.00213,31/31;到 trtllm 差距 3.45x->2.44x

**Why it works**: 双缓冲把 MLP 从 1 tile 提到 2 tile 在飞,把 KV load 延迟藏进 Tensor-Core 计算后面;收益最大恰在 page256 的 16-tile regime(如预期);3-stage(224KB)超 SMEM optin,只能 2-stage(160KB)

<sub>`async_copy` `multi_buffering` `overlap_pipelining` `latency_bound` `smem_lds_capacity`  session `6270b3b2` · commit `2346ae4`</sub>

## v9 tcgen05 qk tmem softmax

**Trigger**: latency_bound, launch_overhead — decode stage (foundation build-up) / single-tile M=64 N=64 D=256 (QK^T decode shape)

**Technique**: 搭 tcgen05 QK^T 单-tile path: 向量化 128-bit cp.async 进 swizzled tcgen05 SMEM -> tcgen05.MmaF16BF16Op(64,64,16) QK^T -> TMEM -> Ld16x64bOp readback -> *scale_log2 -> exp2,单 CTA

**Expected**: 证明 tcgen05 QK^T + TMEM readback + 寄存器侧 softmax numerator 在 SM103 正确,作为全融合的基座

**Effect**: vectorized cp.async+MMA 8.8us(比 scalar 28.5us 快 3.2x, de317cd);K=256 full-shape 11.2us(e263755);QK+scale step1 10.9us(37b3e95);+exp2 step2 11.3us PASS(ac54f15)。单-tile、launch-bound regime

**Why it works**: 证明 tcgen05 QK^T + TMEM readback + 逐元素寄存器算术(scale/exp2)在 SM103 正确;还缺 softmax 归约、P@V、paged gather、多-tile 循环。11.3us 是单 CTA tiny tile 的 launch 主导值,不是可比性能数

<sub>`tcgen05_umma` `tmem` `vectorized_load` `online_softmax` `latency_bound` `launch_overhead`  session `6270b3b2` · commit `ac54f15`</sub>

