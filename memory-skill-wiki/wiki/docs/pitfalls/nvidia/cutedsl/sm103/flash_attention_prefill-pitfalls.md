# Flash Attention Prefill on sm103 — Pitfalls

Traps hit while optimizing `flash_attention_prefill` — CuTeDSL (cutlass 4.5.2), FA4 flash_attn.cute general FlashAttentionForwardSm100, bf16 in/out, fp32 accum, hd256, GQA (qwen3.7-max/3.5-plus), paged KV + varlen + seqused_k on B300 / Blackwell / sm103 / sm100a.
Companion to:

- Optimization journey: [journey](../../../../ref-docs/nvidia/cutedsl/sm103/flash_attention_prefill-optimization.md)
- Optimization highlights: [highlights](../../../../kernel-opt/nvidia/cutedsl/sm103/flash_attention_prefill.md)

---

## 1. hd256 1cta cheap levers null

**Trap**: 在 1-CTA hd256 上试低成本旋钮:(1) 开 CLC persistent 调度(FA_CLC=1,确认走到 scheduling_mode=CLC);(2) 扫 split_P_arrive∈{32,64,96}(env FA4_SPLIT_P);同时 ncu 深挖对 trtllm-gen 的差距（预期 persistent 调度平衡多 tile、split_P 改善 tail-overlap,应缩小对 trtllm 的 SOL 差距）

**Result**: CLC persistent:0.988 vs 0.991ms,对 B=1 prefill 无帮助;split_P_arrive 三档全 0.992-0.998ms,无效。ncu@7680 深 diff:Duration 960 vs trtllm 612us,Compute SOL 46.8 vs 78.8%,tensor active 44.5 vs 78.1%,冒烟枪 = stall_barrier 7.31 vs 0.002 warps/issue-cyc。占用 15.5 vs 18.6%(都低,不是瓶颈)

**Why**: 负向但定位了真瓶颈:CLC persistent 靠平衡众多 tile 起效,B=1 prefill 本就均衡故无用;split_P 调的是 tail-overlap,而瓶颈是 bulk 的 softmax-wait。真因是 q_stage=1(TMEM 512 列:2×S=256+O=256=512,q_stage=2 要 768 放不下)导致没有第二个 Q-tile 可重叠 → MMA 每个 KV 块在 producer_acquire 等 softmax 产 P → barrier stall。关键反直觉:trtllm 同 18.6% 占用却 79% SOL,证明瓶颈是流水/调度效率不是占用

**Lesson**: 负向,但排除了廉价旋钮并锁定真瓶颈。CLC 靠平衡众多 tile 起效,B=1 prefill 本就均衡→无用;split_P 只动 tail,而瓶颈是 bulk 的 softmax-wait。真因:q_stage=1 受 TMEM 512 列硬限(2S+O=512,q_stage=2 需 768),没有第二 Q-tile 可重叠,MMA 每个 KV 块卡在 producer_acquire 等 softmax。最重要的反直觉结论:trtllm 在同样 ~18.6% 占用下拿到 79% SOL(≈我们 1.7x),证明 hd256 1-CTA 的限制器不是低占用,而是流水/调度效率(persistent-context + 更满的 tcgen05 重叠)。规律:占用相近而 SOL 差一大截时,别再调占用/launch/tail 旋钮,直接做流水 overlap 的结构性改造(或换 trtllm-gen)。

<sub>`persistent_kernel` `launch_optimization` `pipeline_stall` `barrier_sync` `latency_bound`  session `4b8ff1d7` · commit `2b1511f`</sub>

## 2. hd256 2cta tmem nan garbage

**Trap**: 把 FA4 hd256 2CTA 内核接进真实 pai-vllm(qwen3.7-max, NVFP4 MoE, TP4)跑端到端;出现全 token-0 乱码后用 dump+replay 逐层排查 NaN 来源（预期 hd256 2CTA 内核在真实 serving 下应产出与离线一致的正确输出）

**Result**: vLLM 里输出全 token-0 乱码;差分探针 out_nan≈19000 而同进程 torch 参考 ref_nan=0,证明是内核写出 NaN;NaN 命中 49 个固定的偶数输出通道、每 (token,head)/每 TP rank 一致(确定性 O-epilogue 腐蚀,D=256);同一编译二进制 + 同 clean 输入在离线任何配置(JIT 缓存开关/单机/NCCL 4 进程/脏 SMEM/NaN 分配器)都 0 NaN。加 V-SMEM flush 后 out_nan 19000->0 但输出仍巨错(max|out-ref|~2e38, 36% 有限值>1e30) → 仍乱码

**Why**: 根因确认为 FA4 hd256 内核在 Blackwell 上的已知 TMEM 容量 bug(Dao-AILab issue #1959):hd256 accumulator tmem_s_offset=0 + tmem_o_offset=256 已填满全部 512 TMEM 列,再叠加 P 与 2CTA 跨 CTA 累加超容 → accumulator 腐蚀 → 输出巨值/NaN。偶数通道 = 2SM MMA 的 peer-CTA 半区,在 vLLM 并发 TMEM 用户(NVFP4 MoE/GDN cutlass)下不可靠。上游 vllm-org 0.23 明确有 guard(head_size>128 且 !=192 在 Blackwell 上强制退回 FA2),pai-vllm 无此 guard 且 VLLM_FLASH_ATTN_VERSION=4 强上 → 腐蚀。V-SMEM flush 只掩盖 NaN 症状、掩不住 accumulator 巨值腐蚀

**Lesson**: 负向,根因 = FA4 hd256 内核在 Blackwell 的已知 TMEM 容量 bug(Dao-AILab #1959)。hd256 的 tmem_s_offset=0 + tmem_o_offset=256 已占满 512 TMEM 列,叠加 P 与 2CTA 跨 CTA 累加超容 → accumulator 腐蚀成巨值/NaN;偶数通道 = 2SM MMA 的 peer-CTA 半区,在 vLLM 并发 TMEM 用户下不可靠(离线唯一 TMEM 用户则干净)。修法:上游 vllm-org 0.23 的做法是 head_size>128&!=192 在 Blackwell 强制退回 FA2;pai-vllm 缺此 guard。真正修好要么 mirror 该 guard,要么改内核不用 2CTA(见 1-CTA 记录)。关键教训:'离线正确、集成乱码' 且腐蚀落在固定偶数通道时,先怀疑 2CTA/peer-CTA 的 TMEM/DSMEM 在并发上下文下的完整性,而非输入或算子二次源;V-SMEM flush 能消 NaN 但消不了 accumulator 超容腐蚀,别被它误导。

<sub>`tmem` `cta_cooperation` `numerics_fix` `numerical_instability` `smem_lds_capacity`  session `4b8ff1d7` · commit `2b1511f`</sub>

## 3. hd256 pingpong compile explosion

**Trap**: Option A' — 把单 softmax-warpgroup 的 S buffer 做 2-stage ping-pong(QK[i+1] 提前写另一 buffer,与 softmax[i] 重叠),消除 barrier stall。三种写法各实现一遍:(1) runtime buf 索引 + if/else 复制 gemm/pipeline;(2) 全 runtime 单 gemm(runtime acc_tmem_addr + runtime pipeline w_index,零复制);(3) constexpr range_constexpr(2) 展开 + 循环外 parity if/else（预期 s_stages=2 的 S 双缓冲让 QK 与 softmax 重叠,把 tensor-core 空转(barrier stall 4.08)降下去,预期 +5-7% SOL(60->~66%)）

**Result**: 三种写法全部在 nvidia-cutlass-dsl 4.5.2 的进程内 MLIR/LLVM 编译爆炸(纯 Python 层 100% CPU、无 ptxas 子进程):if/else 复制 29min 杀掉、全 runtime 7.5min+ 杀掉、constexpr 展开 20min@100%CPU 杀掉。对照:纯 2-stage pipeline SIZING(s_stages=2)+ 普通循环编译很快且 PASS(rel_l2 1.86e-3)

**Why**: 负向:爆炸根因是 QK-ahead 的 cross-buffer commit(nbuf)/acquire(buf) 循环结构本身,4.5.2 编译不了 —— 不是 runtime 索引、不是分支复制、也不是 pipeline sizing(三者已逐一排除,全 runtime 与复制版爆炸方式一致)。q_stage=2(hd128)能编是因其两 stage 是两个独立 Q-tile、成对 per-stage acquire/commit;而单 softmax-warpgroup 的 KV 重叠需要跨 buffer 模式,编译器 choke

**Lesson**: 负向,且经三次确认为死路。爆炸根因是 QK-ahead 的 cross-buffer commit(nbuf)/acquire(buf) 循环结构本身,不是 runtime 索引、不是分支复制、不是 pipeline sizing(全 runtime 版无复制、体积≈默认,仍一样爆)。q_stage=2(hd128)能编,是因其两 stage 为两个独立 Q-tile、成对 per-stage acquire/commit;单 warpgroup 的 KV 重叠需要跨 buffer 模式,4.5.2 无法 legalize。规律:CuTeDSL 4.5.2 里不要尝试单 warpgroup 的跨 buffer ping-pong overlap(commit 一个 buffer 同时 acquire 另一个),~30min 编译周期使迭代不可行;能过的重叠形态是成对独立 stage(q_stage 式)。此结构性 gap 只能靠 DSL 升级或改用 trtllm-gen 解。V1 tile_n=96 即此内核在 4.5.2 下的可达天花板。

<sub>`overlap_pipelining` `multi_buffering` `x-dsl-compile-limit` `pipeline_stall` `barrier_sync`  session `4b8ff1d7` · commit `2b1511f`</sub>

## 4. v1 kv stage pipeline depth

**Trap**: 扫 hd256 2CTA 内核的 KV 流水级数 kv_stage(K/V 多缓冲深度):试 3 / 4 / 5,同时试 ex2_emu_res=3(更低阶多项式)（预期 更深流水(kv_stage=5)本应更好隐藏 TMA 延迟,提升占用/重叠）

**Result**: kv_stage=4 = 最优(57-64% 峰值);kv_stage=3 灾难性 -40%(算力峰值掉到 ~40%);kv_stage=5 略退(~56%,SMEM 变多但换不来 occupancy);ex2_emu_res=3 无改善、略退

**Why**: kv_stage=3 latency-hiding 不足,TMA 加载与 MMA 无法重叠 → 流水饥饿;kv_stage=5 多占的 SMEM 并不能提升 occupancy(已被寄存器/SMEM 双重限死在 18.75%),纯浪费。存在一个明确的甜点:太浅饿死、太深浪费

**Lesson**: 负向教训。kv_stage=3 时 latency-hiding 不足,TMA 与 MMA 重叠不起来导致流水饥饿;kv_stage=5 多吃的 SMEM 换不来 occupancy(内核已被寄存器+SMEM 双重锁在 18.75%),纯浪费。规律:多缓冲深度有明确甜点——太浅饿死流水(代价极大),太深在 occupancy 已饱和时零收益。先找甜点再谈别的,且当 occupancy 已被卡死时,加 SMEM 深度没有意义。

<sub>`multi_buffering` `overlap_pipelining` `pipeline_stall` `smem_lds_capacity` `occupancy_limited`  session `4b8ff1d7`</sub>

