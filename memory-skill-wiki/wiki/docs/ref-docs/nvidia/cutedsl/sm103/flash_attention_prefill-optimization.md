# Flash Attention Prefill on sm103 — Optimization Journey (ref-docs)

**Last Updated**: 2026-07-03  ·  12 recorded attempt(s)

Hardware: B300 / Blackwell / sm103 / sm100a · DSL: CuTeDSL (cutlass 4.5.2), FA4 flash_attn.cute general FlashAttentionForwardSm100 · dtype: bf16 in/out, fp32 accum · shapes: hd256, GQA (qwen3.7-max/3.5-plus), paged KV + varlen + seqused_k

---

## Attempts in detail

### ✅ hd256 1cta fix

工况: B300/sm103,FA4 hd256 bf16 在真实 vLLM(paged+varlen+seqused_k,并发 NVFP4 MoE)下 2CTA 乱码。要一个正确又能 serving 的路径。
方法: 加 env 门 FA4_HD256_1CTA=1,把 hd256 从专用 2CTA 内核改走通用 FlashAttentionForwardSm100 的 1-CTA 模式(use_2cta_instrs=False/CtaGroup.ONE,tile_n=128,q_stage=1;bf16 必须 q_stage=1,否则 Q+O 就要 256KB SMEM 超 224KB 上限)。默认关,不改 vLLM 默认行为。
预期: 1-CTA 每 CTA 在自己 TMEM 算完整 tile,无 peer CTA、无跨 CTA accumulator、无 DSMEM 耦合 → 对并发 TMEM 用户免疫 → 消乱码。
实测: 离线 test_seqused_paged.py 14/14 PASS(PAD_FILL=garbage/zero;PAD_FILL=nan 仅 1/14,因通用内核缺 partial-tile NaN-flush,但 vLLM 的 KV cache 是 torch.zeros,0*0=0 无 NaN,属 harness 假象)。真实 pai-vllm qwen3.7-max TP4 端到端 3 prompt 全连贯,乱码消失。serving 路径 1-CTA 更快:decode 4.2-6.4x、prefill 1.5-2.3x;仅大 dense 单序列 prefill 2-CTA 快 1.1-1.3x。
归因与结论: 正向,是 hd256 乱码的根治方向。CtaGroup.ONE 消掉了 2SM MMA 的 peer-CTA 半区,彻底断开跨 CTA TMEM/DSMEM 相干依赖 → 并发上下文无法腐蚀 accumulator。且它在 serving 更快:decode M=1 时 2CTA 纯浪费 cluster,2CTA varlen 还走慢的 SingleTileVarlenScheduler。规律:Blackwell 上 hd256 若因 TMEM 超容/peer-CTA 在集成中出问题,退到 1-CTA 往往同时更正确且在 decode-heavy serving 更快——2CTA 的价值只在大 dense prefill。注意 bf16 1-CTA 受 SMEM 限必须 q_stage=1。

<sub>`cta_cooperation` `tmem` `tiling` `numerical_instability` `grid_underutilization`  session `4b8ff1d7` · commit `2b1511f`</sub>

### ✅ hd256 1cta tile n96

工况: B300/sm103,FA4 hd256 1-CTA serving prefill(通用 FlashAttentionForwardSm100,bf16,q_stage=1,kv_stage=1 时 199KB SMEM),4 个 biz shape。这是消乱码后的性能优化campaign(shipped V1)。
方法: 把 tile_n 从 128 降到 96(env FA4_TILE_N,后转为默认)。tile_n<128 且 ≠ page_size 会掉出 TMA、改走非-TMA paged gather(cp.async),但每级 KV-SMEM 减半 → 容得下 kv_stage=2 预取 + 腾 TMEM。
预期: kv_stage=2 增加 KV 预取重叠,缓解 MMA 每个 KV 块 barrier-等 softmax 产 P 的停顿。
实测: 4 shape latency 快 12-18%(0.848/0.928/1.014/1.032ms vs 0.99/1.08/1.15/1.21),1119-1167TF,对 trtllm gap 1.52→1.29x / 1.32→1.18x。ncu@7680:Compute SOL 46.8→59.6%,tensor active 44.5→53.9%,barrier stall 7.31→4.08,occ 15.5→19.9%(>trtllm 18.6),dyn smem 198→231KB。14/14 PASS 数值不变;真实 TP4 端到端 12.03 vs 13.04us/call(~8% 更快),token 完全一致。
归因与结论: 正向,是本 campaign 的 shipped 收益。tile_n=96 是仍能给出 kv_stage=2 的最大 tile(112 回退 kv_stage=1,64 变 kv_stage=3 迭代过多反降);kv_stage=2 的预取重叠把 barrier stall 砍近半。规律:当 SMEM 卡住多缓冲深度时,略降 tile_n(即使掉出 TMA 改 cp.async gather)以换 kv_stage+1 常是净正——只要 tile 别小到迭代爆炸。注意占用升到 19.9% 已超 trtllm,但性能仍差——说明占用不是瓶颈(见 trtllm ceiling 记录)。

<sub>`tiling` `multi_buffering` `overlap_pipelining` `pipeline_stall` `barrier_sync` `smem_lds_capacity`  session `4b8ff1d7` · commit `2b1511f`</sub>

### ✅ v1 exp2 freq tuning

工况: B300/sm103,FA4 hd256 2CTA dense prefill,bf16,tile 128x128x256,4 生产 shape。
方法: V1 只改一个调参——软件 exp2 的交织频率 ex2_emu_freq 14→20(_TUNING_CONFIG,hd256 causal),使 softmax warp 的 6 阶 Horner exp2 多项式相对 score 生产计算得更疏,降低与 correction warp 的 FMA 单元争用。
预期: 2-8% 小提升;内核被 18.75% occupancy(每 softmax warp 256 regs + 2CTA 共 384KB SMEM)结构性锁死,只能微调交织平衡。
实测: S7680 -0.3%、S8064 -2.5%、S8320 -0.8%、S8576 -8.2%(最大收益),均值 TFLOPS +3%,精度不变。ncu:SM Throughput 74.98% 但仅 57-64% 峰值,No-Eligible 61.36%,首要 stall = L1TEX scoreboard(39.1%),grid(64,32)×384。
归因与结论: 正向但幅度很小。机制成立(exp2 的 FMA 与 correction 抢单元,降频给 MMA 更连续窗口)。但真正的天花板是结构性的:12 warp 中仅 1/12 做 MMA、软件 exp2 的 FMA 不进 FLOP 预算、L1TEX 39% stall 是 TMA↔MMA 流水气泡。教训:warp-specialized FA 内核里,exp2 交织频率是可安全微调的 3% 级旋钮,但触不到占比更大的流水气泡与低 occupancy——要大提升必须做结构性流水解耦。

<sub>`fast_math` `mma_scheduling` `compute_bound` `pipeline_stall` `occupancy_limited`  session `4b8ff1d7`</sub>

### ✅ v2 sm103 hw exp2

工况: B300/sm103,FA4 hd256 2CTA dense prefill,bf16,4 生产 shape。SM103(B300)的硬件 SFU exp2 比 SM100 快。
方法: 给 hd256 专用内核补上 SM103 架构检测(BaseDSL 取 arch enum,is_family_of(sm_103f)),在 SM103 上把软件 exp2(14 系数 Horner)关掉(ex2_emu_freq=0),改用硬件 SFU exp2。FA4 非-hd256 内核本已这么做,但 hd256 内核硬编码 is_sm103=False,一直没享受到。
预期: 对 softmax 计算占比大的长 shape(S≥8320)有 10-20% 提升;SM103 上软件模拟成了多余的 FMA 开销,走硬件 SFU 能把 FMA 让给 correction。
实测: A/B 公平比较(同 seed/warmup/rep):S8320 926→782us(+15.5%),S8576 961→824us(+14.3%),S7680 不变(小 shape exp2 占比可忽略),S8064 -1.3% 噪声;精度 rel_l2~0.002 不变。
归因与结论: 正向。软件 exp2 是为绕过 SM100 的 SFU 瓶颈而设,但在 SFU 更快的 SM103 上它反而是纯开销——关掉它让 softmax 更快、流水气泡更小。规律:跨代移植内核时,针对旧架构瓶颈的 workaround(如软件 exp2)在新架构上可能变成负担,要按 arch 动态开关;且此类收益随 softmax 计算占比放大(长 seqlen 才明显)。FA4 官方知道这点却漏给了 hd256 专用内核——移植时留意专用内核常滞后于通用内核的架构优化。

<sub>`fast_math` `mma_scheduling` `compute_bound` `pipeline_stall`  session `4b8ff1d7` · commit `2b1511f`</sub>

### ✅ vllm fa4 build

工况: 把 vllm 与 fa4(vllm_flash_attn = FA2 sm80/8.9 + FA3 sm90 CUDA 内核)源码 editable 编进 conda env xingze(py3.12,torch 2.11.0+cu130,nvcc 13.2),B300 机器,构建 CPU 被 cgroup 限到 ~1 核。
方法: vllm 先 use_existing_torch.py 剥掉 torch 版本钉,再 pip install -e . --no-build-isolation;fa4 用 pip install -e . --no-build-isolation --no-deps。
预期: 两包装好且不动现有 torch/CUDA 栈。
实测: 四个坑——(1) vllm 要 setuptools>=77(env 70.2.0 在 PEP-639 license 行失败),升到 80.10.2;(2) fa4 缺 --no-deps 会装 metadata 钉的 torch==2.4.0(797MB)覆盖 torch 2.11,必须 --no-deps;(3) vllm ~76 target ~50min、fa4 251 target ~3h(1 核 quota,MAX_JOBS 无效);(4) test_fa4_smoke 失败,因 bundled flash_attn 的 __init__.py 硬 import legacy flash_attn_2_cuda(本仓库不构建),需 try/except 包住才可用 flash_attn.cute。
归因与结论: 正向(构建成功),但四个坑都可复用:pip editable 装钉死旧 torch 的包时永远加 --no-deps 防覆盖 CUDA 栈;新版 pyproject 的 SPDX license 需 setuptools≥77;B300 上 flash_attn 父包会贪婪 import 只在 PAI wheel 里的 flash_attn_2_cuda,用 try/except 隔离即可放行 CuTeDSL FA4 路径;1-核 quota 下别指望 MAX_JOBS,预留 3h+。

<sub>`x-build-integration` `x-build-integration`  session `4b8ff1d7`</sub>

### ✅ vllm hd256 seqused integration

工况: B300/sm103,把 FA4 hd256 接进 vLLM 的 paged+varlen 注意力路径(fa_utils.py 的 FA4 wrapper),GQA,page_size==128,block_table row-major。
方法/问题: vLLM 的 FA4 wrapper 总给内核传 seqused_k(fa_utils.py:145,注释假定 '内核靠 seqused_k 界定 KV walk'),而 FA4 baseline hd256 2CTA 内核硬 assert 拒绝 seqused_q/seqused_k → warmup 即崩。
预期: 让 hd256 在 vLLM 变长 paged 路径跑通。
实测: 两条解法。权宜(仅 prefill):把 seqused_k 从 wrapper 签名里删掉,使 has_seqused_k=False(该标志由 inspect.signature 探测),vLLM 改传 cu_seqlens_k(hd256 支持),max_tokens=1 prefill 的 KV 由 cu_seqlens_k 完整描述 → 跑通抓到正确 trace。根治:fa4_bf16_varlen commit 2b1511f 给 hd256 内核补 seqused_q/k(kernel 参数 + 四处 warp 段 per-batch 覆写 seqlen_k=seqused_k[batch_coord],经 get_trip_start_count_via_block_info 真正截断 KV 迭代),前向 rel_l2≈0 零回归;但 backward 仍 assert 拒绝。
归因与结论: 正向(集成打通)。关键规律:vLLM 假设所有 FA 内核用 seqused_k 界定 KV walk,这对 hd≤128 成立、对 hd256 专用内核不成立 → decode(变长只由 seqused_k 编码)根本跑不了,只有前向 prefill 能用 cu_seqlens_k 绕过。要真正 serving 必须给 hd256 内核补 seqused_k 支持(让 seqlen_k 流入 trip-count 计算),且注意 page_size 必须==128(TMA-only)、block_table 需 row-major 连续。debug 此类崩溃先看 vLLM 用 signature 探测决定传 seqused_k 还是 cu_seqlens_k。

<sub>`paged_gather` `x-varlen-support` `x-build-integration`  session `4b8ff1d7` · commit `2b1511f`</sub>

### ➖ baseline v0 dense

工况: B300/sm103 上 FA4 CuTeDSL(cutlass 4.5.2)hd256 专用 2CTA 前向 dense prefill,bf16 in/out + fp32 softmax,GQA 32Q/2KV,4 个生产 shape S∈{7680,8064,8320,8576}。
方法: 直接用 pip 安装的 flash_attn.cute(sm100_hd256_2cta_fmha_forward,tcgen05.mma+TMEM+TMA+软件 exp2+warp specialization)作 V0 基线,do_bench 测延迟,不改任何代码。
预期: roofline 上 AI=3614-4038 远超 ridge=281,判为强 compute-bound;停机目标 90% 峰值 = 2025 TFLOPS / 477-595us/shape。
实测: S7680 668us(1446TF,64.3%峰值),S8064 829us(1286TF,57.1%),S8320 880us(1289TF,57.3%),S8576 946us(1274TF,56.6%);rel_l2 max 0.00199,4/4 PASS。较生产 paged 基线(bladnn_fa4,2725-3364us)快 3.6-4.1x,但仅达算力峰值 57-64%。
归因与结论: compute-bound 成立;距峰值 35-40% 的差距源于 warp-specialized 内核的固有 18.75% occupancy(每 CTA 12 warp、大寄存器/SMEM 预算)与生产者/消费者流水气泡,而非带宽。基线可用但离天花板远,后续应针对流水气泡与 exp2 计算而非访存下手。dense 接口只是能力上界参考——真正 serving 需 paged+varlen(见 seqused 集成记录)。

<sub>`tcgen05_umma` `tmem` `warp_specialization` `compute_bound` `occupancy_limited`  session `4b8ff1d7` · commit `2c839c3`</sub>

### ➖ hd256 vs trtllm gen ceiling

工况: B300/sm103,hd256 GQA(Hq32/Hkv2)paged causal prefill,4 biz shape。用 flashinfer 0.6.12 内置的 TRT-LLM trtllm-gen FMHA 作对标(无需自建 ~10GB TRT-LLM 框架)。
方法: 跑 trtllm_batch_context_with_kv_cache(paged block_tables+seq_lens),ncu --set full 测 SOL/latency,与 FA4 各形态(dense 2CTA、1-CTA varlen tile_n=96、varlen 2CTA)横比。
预期: 确定 FA4 hd256 在可用 paged 路径的天花板,决定自研还是采用 trtllm-gen。
实测: trtllm-gen native kernel(fmhaSm103a...H256PagedKvCausalP64...PersistentContext):Compute SOL 78.8-80.6%,Duration 612-765us,~1300-1478TF,occ 18.6%,rel_l2~1.9e-3。对比 FA4 dense 2CTA 0.69-0.87ms/74-76%(serving 不可用)、1-CTA varlen 0.80-0.99ms/60%、varlen 2CTA 1.2-1.48ms/44%。trtllm-gen 既最快又在可用 paged 路径,比 1-CTA 快 ~1.3-1.5x,SOL 甚至高于 FA4 离线 dense。
归因与结论: 中性(对标/选型)。trtllm-gen 用 persistent-context 调度 + 更满的 tcgen05 重叠,同 18.6% 占用达 79% SOL;FA4 1-CTA 因 CuTeDSL 4.5.2 编不出 ping-pong overlap 卡在 60%。规律:在 Blackwell 上 hd256 paged serving,flashinfer 打包的 trtllm-gen native cubin 是现成天花板(原生变长、零内核风险),优先直接采用;自研 CuTeDSL 内核作正确性 fallback。选型时先测 flashinfer 里的 trtllm-gen 再决定是否投入内核开发。

<sub>`persistent_kernel` `tcgen05_umma` `overlap_pipelining` `compute_bound` `pipeline_stall`  session `4b8ff1d7` · commit `2b1511f`</sub>

### ❌ hd256 1cta cheap levers null

工况: B300/sm103,FA4 hd256 1-CTA serving prefill(bf16,q_stage=1),biz S=7680,与 trtllm-gen 对标。
方法: 先试低成本旋钮——开 CLC persistent 调度(FA_CLC=1,已确认真的走 CLC),扫 split_P_arrive∈{32,64,96};并用 ncu 深挖对 trtllm 的差距来源。
预期: persistent 调度平衡多 tile、split_P 改善 tail-overlap,应能缩差距。
实测: CLC 0.988 vs 0.991ms(无用),split_P 三档 0.992-0.998ms(无用)。ncu 深 diff:Duration 960 vs 612us,Compute SOL 46.8 vs 78.8%,tensor active 44.5 vs 78.1%,冒烟枪 stall_barrier 7.31 vs 0.002,占用 15.5 vs 18.6%(两边都低)。
归因与结论: 负向,但排除了廉价旋钮并锁定真瓶颈。CLC 靠平衡众多 tile 起效,B=1 prefill 本就均衡→无用;split_P 只动 tail,而瓶颈是 bulk 的 softmax-wait。真因:q_stage=1 受 TMEM 512 列硬限(2S+O=512,q_stage=2 需 768),没有第二 Q-tile 可重叠,MMA 每个 KV 块卡在 producer_acquire 等 softmax。最重要的反直觉结论:trtllm 在同样 ~18.6% 占用下拿到 79% SOL(≈我们 1.7x),证明 hd256 1-CTA 的限制器不是低占用,而是流水/调度效率(persistent-context + 更满的 tcgen05 重叠)。规律:占用相近而 SOL 差一大截时,别再调占用/launch/tail 旋钮,直接做流水 overlap 的结构性改造(或换 trtllm-gen)。

<sub>`persistent_kernel` `launch_optimization` `pipeline_stall` `barrier_sync` `latency_bound`  session `4b8ff1d7` · commit `2b1511f`</sub>

### ❌ hd256 2cta tmem nan garbage

工况: B300/sm103,FA4 CuTeDSL hd256 2CTA 前向,bf16,接入真实 pai-vllm(qwen3.7-max NVFP4 MoE,TP4 多进程,paged+varlen+seqused_k,与 NVFP4 MoE/GDN 的 cutlass kernel 并发)。
方法: 端到端跑出全 token-0 乱码后,用 FA4_DUMP_PATH 抓 vLLM 首次 FA4 调用、离线 replay,并用同进程差分探针定位 NaN 来源。
预期: 内核在真实 serving 下应和离线一致产出正确输出。
实测: 输入全 clean(q/被 attend 的 KV/torch 参考均 0 NaN);同一编译二进制在离线任何配置(JIT 缓存开关、单机、NCCL 4 进程、脏 SMEM、NaN 毒化分配器)都 0 NaN;但 vLLM 运行时内核写出 NaN(out_nan≈19000,ref_nan=0),命中 49 个固定偶数输出通道、每 rank 一致。加 both-CTA V-SMEM flush 后 out_nan→0,但输出仍巨错(max|out-ref|~2e38,36% 有限值>1e30),仍乱码。
归因与结论: 负向,根因 = FA4 hd256 内核在 Blackwell 的已知 TMEM 容量 bug(Dao-AILab #1959)。hd256 的 tmem_s_offset=0 + tmem_o_offset=256 已占满 512 TMEM 列,叠加 P 与 2CTA 跨 CTA 累加超容 → accumulator 腐蚀成巨值/NaN;偶数通道 = 2SM MMA 的 peer-CTA 半区,在 vLLM 并发 TMEM 用户下不可靠(离线唯一 TMEM 用户则干净)。修法:上游 vllm-org 0.23 的做法是 head_size>128&!=192 在 Blackwell 强制退回 FA2;pai-vllm 缺此 guard。真正修好要么 mirror 该 guard,要么改内核不用 2CTA(见 1-CTA 记录)。关键教训:'离线正确、集成乱码' 且腐蚀落在固定偶数通道时,先怀疑 2CTA/peer-CTA 的 TMEM/DSMEM 在并发上下文下的完整性,而非输入或算子二次源;V-SMEM flush 能消 NaN 但消不了 accumulator 超容腐蚀,别被它误导。

<sub>`tmem` `cta_cooperation` `numerics_fix` `numerical_instability` `smem_lds_capacity`  session `4b8ff1d7` · commit `2b1511f`</sub>

### ❌ hd256 pingpong compile explosion

工况: B300/sm103,FA4 hd256 1-CTA(通用内核)serving prefill,DSL = nvidia-cutlass-dsl 4.5.2。V1(tile_n=96,60% SOL)之上想再逼近 trtllm 的 79%,唯一剩的结构性 gap 是 barrier stall(MMA 每个 KV 块等 softmax 产 P)。
方法: Option A'——单 softmax-warpgroup 的 S buffer 做 2-stage ping-pong,QK[i+1] 提前写另一 buffer 与 softmax[i] 重叠。三种代码策略各实现:if/else 复制(constexpr 索引)、全 runtime 单 gemm(runtime acc_tmem_addr + runtime pipeline w_index、零复制)、constexpr range_constexpr(2) 展开。
预期: S 双缓冲让 QK 与 softmax 重叠,消 barrier stall(4.08→~0),预期 +5-7% SOL。
实测: 三种写法全部撞同一堵墙——4.5.2 进程内 MLIR/LLVM 编译爆炸(纯 Python 100% CPU、无 ptxas 子进程):if/else 29min、全 runtime 7.5min+、constexpr 展开 20min,全部杀掉。对照实验关键:纯 2-stage pipeline SIZING(s_stages=2)+ 普通循环编译很快且 PASS(rel_l2 1.86e-3)。
归因与结论: 负向,且经三次确认为死路。爆炸根因是 QK-ahead 的 cross-buffer commit(nbuf)/acquire(buf) 循环结构本身,不是 runtime 索引、不是分支复制、不是 pipeline sizing(全 runtime 版无复制、体积≈默认,仍一样爆)。q_stage=2(hd128)能编,是因其两 stage 为两个独立 Q-tile、成对 per-stage acquire/commit;单 warpgroup 的 KV 重叠需要跨 buffer 模式,4.5.2 无法 legalize。规律:CuTeDSL 4.5.2 里不要尝试单 warpgroup 的跨 buffer ping-pong overlap(commit 一个 buffer 同时 acquire 另一个),~30min 编译周期使迭代不可行;能过的重叠形态是成对独立 stage(q_stage 式)。此结构性 gap 只能靠 DSL 升级或改用 trtllm-gen 解。V1 tile_n=96 即此内核在 4.5.2 下的可达天花板。

<sub>`overlap_pipelining` `multi_buffering` `x-dsl-compile-limit` `pipeline_stall` `barrier_sync`  session `4b8ff1d7` · commit `2b1511f`</sub>

### ❌ v1 kv stage pipeline depth

工况: B300/sm103,FA4 hd256 2CTA dense prefill,bf16,4 生产 shape;调 KV 加载流水深度 kv_stage。
方法: 在 V1 调参里扫 kv_stage∈{3,4,5}(K/V 多缓冲级数),另试更低阶软件 exp2 多项式 ex2_emu_res=3。
预期: 更深的 KV 流水应更好隐藏 TMA 加载延迟。
实测: kv_stage=4 是甜点(57-64% 峰值);kv_stage=3 灾难性回退约 -40%(掉到 ~40% 峰值);kv_stage=5 反而略差(~56%);ex2_emu_res=3 无改善甚至略退。
归因与结论: 负向教训。kv_stage=3 时 latency-hiding 不足,TMA 与 MMA 重叠不起来导致流水饥饿;kv_stage=5 多吃的 SMEM 换不来 occupancy(内核已被寄存器+SMEM 双重锁在 18.75%),纯浪费。规律:多缓冲深度有明确甜点——太浅饿死流水(代价极大),太深在 occupancy 已饱和时零收益。先找甜点再谈别的,且当 occupancy 已被卡死时,加 SMEM 深度没有意义。

<sub>`multi_buffering` `overlap_pipelining` `pipeline_stall` `smem_lds_capacity` `occupancy_limited`  session `4b8ff1d7`</sub>

