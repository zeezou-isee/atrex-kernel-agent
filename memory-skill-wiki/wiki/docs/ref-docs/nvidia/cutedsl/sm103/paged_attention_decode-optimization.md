# Paged Attention Decode on sm103 — Optimization Journey (ref-docs)

**Last Updated**: 2026-07-03  ·  27 recorded attempt(s)

Hardware: B300 / Blackwell / sm103 · DSL: CuTeDSL (cutlass 4.5.2) · dtype: bf16 in/out, fp32 partial/accum · shapes: hd256 (D=256), GQA, split-KV num_splits, combine 归约

---

## Attempts in detail

### ✅ late combine kernel ndchunk occupancy

工况: B300/sm103,CuTeDSL(cutlass 4.5.2)paged decode 的 flash-decoding combine(把 split-KV 的 partial 归约成 O),hd256/D=256,bf16 输出、fp32 partial。移植自 V5 combine。
方法: combine 独立成 kernel,grid=(B,nkv,N_DCHUNK):phase1 一线程/行算跨 split 的 global-max + 每 split rescale 因子 + inv 存 smem;phase2 全线程按 D-chunk 加权累加、写 bf16 O(GQA scatter)。以 N_DCHUNK 作占用率旋钮扫参。
预期: combine work 很小、64 CTA 占用率不足,增大 N_DCHUNK 拆更多 CTA 填满 SM 应线性提速直到饱和。
实测: combine 26.65µs(N_DCHUNK=4/64CTA)→18.30(8)→12.43(16)→10.34µs(32/512CTA);N_DCHUNK≥32 平台在 ~10.3µs。full pipeline 53.31→43.15µs,31/31 shape PASS rel≈0.0021。
归因与结论: 正向。小-work 归约核的耗时由占用率主导,N_DCHUNK 拆 D 维提 CTA 数直到 SM 填满即饱和(≥32 无进一步收益)。关键坑:N_DCHUNK 必须整除 D=256——48→DC=5 只覆盖 240 列是静默正确性 bug;取 32(DC=8)。教训: 归约/combine 这类轻核先按占用率调 grid;拆维度旋钮必须整除被拆维度,否则漏尾静默错。

<sub>`occupancy_tuning` `split_k` `online_softmax` `occupancy_limited` `grid_underutilization`  session `3e24042c` · commit `2b76d06`</sub>

### ✅ late kstages2 desqn smem optin

工况: B300/sm103,CuTeDSL warp-spec 分页 decode(a3),shape0 klen=256(4 tile),hd256/M_TILE=128,16 CTA。
方法: 去掉 sQn(64KB Q staging buffer)——Q 改为每个 compute 线程直接从 gmem scalar-gather 自己那行到寄存器,再 relayout 进 swizzled sQ;省出的 smem 用来把 K/V load 从单缓冲开到 KSTAGES=2 双缓冲。
预期: 省 64KB smem 让双缓冲放得下,load(t+1) 与 compute(t) 重叠藏载入延迟;顺带消掉 sVn 别名 sQn 的 Q-pack race。
实测: shape0 decode-path 57.7µs → 30.6µs(1.9×,首次超过 stage4 的 43µs),rel=0.002102 PASS。
归因与结论: 正向。双缓冲重叠 + Q 寄存器直取(免 scalar pack)双重收益,即便仍只有 16 CTA。关键坑:CuTeDSL launch 的 smem 请求必须调到 SM100 max optin=232448 字节(实际需 ~229376);之前设 228000 < 需求,编译不报错、运行时才 CUDA illegal memory access(静默 OOB)。教训: Blackwell 上把 smem 顶到 cudaDevAttrMaxSharedMemoryPerBlockOptin(本机 232448);smem 请求略小于实际用量不会编译报错,只在运行时崩,遇到不明 illegal access 先核对 smem 请求值。

<sub>`multi_buffering` `async_copy` `smem_lds_capacity` `pipeline_stall` `latency_bound`  session `3e24042c` · commit `f68fb22`</sub>

### ✅ late splitkv occupancy beats overlap

工况: B300/sm103,CuTeDSL warp-spec 分页 decode,shape0 klen=256 B=8,hd256/GQA。a4 = a3 的 warp-spec decode 写 partial(O_unnorm/run_max/run_sum)+ 复用 stage4 combine,grid=(B,nkv,num_splits)。
方法: 实测三档 num_splits:NS=4(1 tile/split,64 CTA)、NS=2(2 tile/split,32 CTA,有跨-tile 重叠)、NS=1(4 tile/split,16 CTA,≈a3),用数据决定'高占用率'与'跨-tile 重叠'谁更划算。
预期: 对小 klen 找最优 NS。
实测: shape0 NS=4=18.56µs(最佳)、NS=2=22.5µs、NS=1=32µs;分解 decode 15.3µs + combine 10.5µs = full ~19µs。整体轨迹 stage4 43 → a3(KSTAGES2)30.6 → a4(splitKV)18.56µs,逼近 trtllm 12.3µs(差 1.5×)。klen 256/1024 都 PASS。
归因与结论: 正向。对这个小 klen,提占用率(splitKV 到 ~64 CTA)决定性胜过跨-tile 重叠——因为 KSTAGES=2 的重叠需每 CTA ≥2 tile,而小 klen 下'高占用率'与'多 tile/CTA 重叠'二选一。这正是 trtllm 的路子:占用率靠 splitKV(1 tile/split)、单-tile 内延迟靠多 warp 藏,而非靠跨-tile 重叠。教训: 小-workload decode 先用 splitKV 把 CTA 数堆到 ~SM 数(64),再谈藏延迟;跨-tile 软件流水只对大 klen(多 tile/CTA)有用。

<sub>`split_kv` `grid_underutilization` `occupancy_limited` `latency_bound`  session `3e24042c` · commit `f68fb22`</sub>

### ✅ late v10 readback dst shape bug

工况: B300/sm103,CuTeDSL(cutlass 4.5.2)tcgen05+TMEM 的 paged split-KV decode,bf16 in/out、fp32 softmax/累加,hd256/GQA/q_per_seq=4/block_n=64。接手上一轮卡住的 'PV→O=0' bug。
方法: 不再信任上一轮'多-CTA/PV 出 0'的诊断,改用分段隔离:用一条独立且已验证的 Ld32x32 逐行读路径直接读 TMEM_O,再逐段验证 read / write-chain / tensor 绑定。
预期: 定位 O 全 0 的真根因,让 split-KV decode 端到端正确。
实测: rel_l2 从 1.000000 FAIL 修到 0.002025 PASS;推广到全 31/31 shape PASS(rel≈0.002)。用 Ld32x32 读 O-TMEM 得 sumsq=32.14(非零)直接证明 PV 正确;stage3a 单-CTA 对照仍 21µs PASS 排除环境变化。
归因与结论: 正向。真根因=O 的 readback 寄存器张量按'源(TMEM)分片 shape'创建而非'目的(gmem)分片 shape',Ld16x64 epi-copy 因而静默读回 0(与通过的 stage3a 唯一差异)。上一轮所有 probe(entry-write/gOp scalar-write)都被紧随其后的 readback 覆盖(readback 写满整个 mOp[0,0,0]),看起来像'绑定坏了 / PV=0'。另修两处 latent bug:padding 行(m_idx≥m_rows)全被 mask → tile_max=-inf → exp2(-inf-(-inf))=NaN,用 fmax(tile_max,-1e30) 夹住;以及 V 转置在调试期被删(natural V 得到 P@Vᵀ,rel=1.39),恢复转置载入(sV 物理存 Vᵀ)。教训: tcgen05 readback 的寄存器 fragment 必须按目的张量 shape 建,不能按源;调试'输出全 0/全错'先用一条独立读路径确认上游真值,再逐段隔离,避免被会覆盖同一 buffer 的 readback 误导。

<sub>`tmem` `numerics_fix` `in_kernel_fusion` `numerical_instability`  session `3e24042c` · commit `2b76d06`</sub>

### ✅ late warpspec decode a0 a3 build

工况: B300/sm103,CuTeDSL(cutlass 4.5.2)从零搭 warp-specialized tcgen05 分页 decode,hd256/D=256/GQA/q_per_seq=4/block_n=64,klen 256(4 tile)与 1024(16 tile)。
方法: 类比 stage1→4 分步(a0→a3)。a0=专用 LOAD warp(cp.async 生产者)+ MMA warp(tcgen05 消费者)+ 手写 mbarrier 2-stage 双缓冲的 QK 骨架;a1=把 QK→online-softmax→PV→O 整条 mainloop 交织进 COMPUTE group(warp0-3),load(t+1) 与 compute(t) 重叠;a2=扩到真实 D=256,用 per-D-chunk Ld32x32/St32x32 做 TMEM-O rescale;a3=接真实分页输入(GQA Q-pack + paged gather K/V + V 转置 + causal-tail),对 reference.py 验证。全程 constexpr 循环(绕开动态循环 TMEM 限制)。
预期: 先证明 warp-specialize 机器 + tcgen05 共存与多 tile 数值正确,作为 16-warp 重写地基。
实测: a0 last-tile rel=0.0 PASS 且明显区别于 tile0-2(证明双缓冲把正确 K 交付 MMA);a1 4-tile rel=0.0014;a2 D=256 rel=0.0011;a3 klen=256 rel=0.002102、klen=1024(16 tile)rel=0.002105 均 PASS,16-tile constexpr 约几分钟编译不爆 IR(warp-spec 精简 per-tile body)。
归因与结论: 正向。沉淀多条可迁移 CuTeDSL 规律:(1)专用 load warp 的 cp.async copy 必须按其实际 32 线程 tiled + lane=tidx%32 切片,按 160 线程 tiled 会导致 K/Q 没载全(症状 nonzero_rows=112);(2)tcgen05 tmem-copy atom 不能跨 scf region 边界(动态 for 或 warp-specialize if 都不行)、不能从一个 tensor 建后 partition_S 到别的偏移、不支持闭包捕获 → 必须在使用它的 region 内就地、逐偏移创建;(3)warp-spec 下 compute group 内部同步用 NamedBarrier(128) 而非 __syncthreads(否则与 load warp 死锁);(4)D=256 TMEM-O rescale 用 per-D-chunk Ld32x32(1 线程/行)让 corr 天然按行在每线程手里,绕开 sScale 跨线程广播,第一 tile 特判 ACCUMULATE=False 避 NaN×0;(5)别名 smem(sVn 复用 sQn)在 warp-spec 下必须 barrier 保时序——load warp 的 tile-0 V-gather 覆盖了 compute 还在读的 Q,症状是所有 query/head 均匀 ~0.245 误差(覆盖类 bug 的典型信号),修法把 Q-pack 移到 load/compute 分流前 + sync_threads()。教训: 覆盖类 bug 看'误差是否在所有行均匀';warp-spec 下每条 tmem-copy atom 就地创建、内部同步用 NamedBarrier、别名 smem 加 barrier。

<sub>`warp_specialization` `multi_buffering` `async_copy` `paged_gather` `tmem` `online_softmax` `pipeline_stall` `latency_bound` `numerical_instability`  session `3e24042c` · commit `f68fb22`</sub>

### ✅ pv v orientation rule

工况: B300/sm103,CuTeDSL tcgen05 FMHA 的 PV(P@V)阶段,bf16、先 D=64 探针再 hd256。
方法: 设计确定性最小诊断 —— 令 V[n,d]=d+1(V[0,:] 随 d 变、V[:,0] 恒定,极易区分),P 只选 key 0,则正确 O[m,:]=V[0,:]=[1,2,3...],转置则 O[m,:]=V[:,0]=[1,1,1...];直接读 O 判定,并对 b_major=K 与 MN 各测。
预期: 把 PV 的 V-operand 朝向彻底定死,让 Stage 3c 的 paged gather 能正确写 sV(这是卡死 3b 的谜题,之前多轮盲试)。
实测: 定论并验证 —— tcgen05 PV MMA 计算 P @ (sV)^T,与 b_major 无关(K 与 MN 都给 P@V^T)。喂 V 自然 -> O=[1,1,1...]=P@V^T(错);喂 V^T 或 stride 交换的转置视图 sVn_T 作源 -> O=[1,2,3...]=P@V(对,rel 0.0018)。
归因与结论: 正向,干净可复用规则。根因: autovec_copy 用相同 TV 布局是按物理序拷贝(等价恒等,不做逻辑转置),之前 3b 三种 b_major/转置组合都仍是 P@V^T 就是这个原因;真正的转置需要 stride 交换的源视图 sVn_T、ldmatrix.trans 或 TMA,cp.async 做不了转置。可迁移规则: PV 的 tcgen05 MMA 算 P@(sV)^T,朝向由 sV 物理布局决定而非 b_major —— 要 P@V 就必须让 sV 物理存 V^T(gather 时把每个 key 的 D 值按转置写入 sV,或用 TMA)。方法论: 布局朝向类谜题用确定性构造探针(值可区分 + 已知答案)一次定死,别盲试 b_major。

<sub>`swizzle` `tcgen05_umma` `x-operand_layout` `uncoalesced_access`  session `6270b3b2`</sub>

### ✅ softmax threadlocal reduction

工况: B300/sm103,CuTeDSL tcgen05 in-kernel FMHA,M=128 N=64,多-tile klen=128/256(D=64 隔离),bf16、fp32 累加。
方法: 在 TMEM readback 的 S fragment 上做 online softmax;先打印 Ld32x32bOp 在 (128,64) S tile 上给每线程的 fragment 布局,再决定行归约做法。
预期: 确定 row max/sum 归约方式,跑通多-tile online softmax(running max/sum + correction + 寄存器 O 累加)。
实测: 关键发现 —— Ld32x32bOp 读 (128,64) S 时,128 线程每个恰好持有一整行的 N=64 值(rS=64 elem,128x64=M*N),所以 softmax 行归约与 per-thread O correction 完全线程内、无需跨线程 shuffle。多-tile online softmax 2-tile 与 4-tile 均 rel_l2约0.001 PASS。
归因与结论: 正向,重要简化。根因: N=64 的 tile 宽度让每行恰好落一个线程,running-max/running-sum/correction 全是 per-thread 标量,比 FA4 SoftmaxSm100 的 warp-specialized 跨 lane 归约简单得多。可迁移规则: 选 N=64 tile 时,读回布局让每线程持整行,softmax 变纯线程内运算,免掉 quad/butterfly shuffle —— N=64 是甜点。配套: correction O=O*corr+O_t 在 D=64 可在寄存器做,但 D=256 因 O 超寄存器必须转 TMEM-O rescale(见 hd256 TMEM 预算记录)。

<sub>`online_softmax` `warp_reduction` `tmem` `latency_bound`  session `6270b3b2`</sub>

### ✅ v10 inkernel pv fusion cracked

工况: B300/sm103,CuTeDSL(4.5.2),单-tile M=128 N=64 D=64 隔离验证,bf16 in、fp32 累加,基于 9a14c6d(北园 V10 clean base)的存档外工作副本 kernel_tcgen05/。
方法: 攻克北园 V10 从未跑通的 in-kernel tcgen05 PV 融合 —— (1) S/P/O 放 DISJOINT TMEM 列区(S[0,64)、P bf16[64,96)、O[128,384)),不共用一块;(2) M-tile 对齐 128 用 make_trivial_tiled_mma 得干净 fragment(M=64 会得复合 (16,4),Ld32x32/St32x32 atom 拒绝);(3) P 用 row-bridge 写回 TMEM: Ld32x32bOp 读 S、寄存器 exp2、把 bf16 P 打包成 fp32-word(rP_bf16 与 rP_f32 共享同一寄存器)用 St32x32bOp(Float32) 存;P 作 PV A-operand 用 fp32 的 S iterator + bf16 A-layout 读回;(4) PV MMA 走 TS-mode、a_major=K。
预期: 解掉 V10 记录的两个 bug(TMEM 复用/重置 -> rel_l2=1.5、PV 坐标映射),让单-tile in-kernel FMHA 正确。
实测: QK->softmax->PV->O 全链路 rel_l2=0.0018 PASS(v1 unnorm,v2 加真 softmax 仍 0.0018);do_bench 11.2us 但 LAUNCH-BOUND(单 CTA tiny tile,约等 QK-only 11.4us,GPU 计算 <1us)。
归因与结论: 正向,重大突破(北园 V7-V10 多轮未通)。三条根因即三条可迁移规则: (a) TMEM 里 S 和 O 必须占不相交列区、O 循环前只清零一次并累加 —— ACCUMULATE=False 只重置下一次 MMA 的首个 k-block,不能清跨-tile 累加区,这正是 V10 的 rel_l2=1.5;(b) tcgen05 的 32x32b copy atom(Ld32x32/St32x32)要求干净 128 行 fragment,M<128 得到复合 (16,4) fragment 会不匹配,用 M=128 + make_trivial_tiled_mma;(c) 造 P 的 bf16 A-operand 不能对 TMEM 指针 recast_ptr(bf16)(会翻倍 M-stride 131072 vs 65536),要用 fp32 的 S fragment iterator 配 bf16 A-layout。计量提醒: 11.2us 是 launch-bound,此版价值是正确性突破而非速度。

<sub>`tcgen05_umma` `tmem` `in_kernel_fusion` `latency_bound` `register_pressure`  session `6270b3b2`</sub>

### ✅ v1 vectorized gather

工况: B300/sm103,CuTeDSL paged decode attention,bf16、hd256、GQA、q_per_seq=4、block_n=64、block_size {64,256}。
方法: 用向量化 128-bit 合并 KV gather 替换 V0 的逐元素 scalar gather —— 依赖 BLOCK_N=64 且 block_size in {64,256} 的不变量(每 64-key tile 恰落单一物理块、head_dim 连续),做单次 (N,D) block-slice、32 线程 x 8 bf16/行。
预期: 干掉 scalar gather 的 uncoalesced 访存(V0 的 long_scoreboard 9.73 主导),把 memory_bound 拉下来。
实测: mean ~196us(shape6 222.7us),相对 V0 ~2400us 提速约 4-10.8x;big-shape 543 GB/s;long_scoreboard 9.73 大幅塌缩;rel_err ~0.002 与 V0 一致,31/31 PASS。
归因与结论: 正向。合并 128-bit load 把 LD 数量降约 8x,直接消除 uncoalesced-access 这个 V0 主瓶颈。代价是暴露新瓶颈:瓶颈从 memory_bound 迁到 latency_bound —— 小 grid(waves/SM 0.22,B*nkv << 148 SMs)、~6.3% occupancy、235 regs、load 与 MMA 无重叠。规则: 单一物理块不变量是能做单次合并 gather 的关键前提;向量化访存是 memory-bound decode 的第一杠杆,但收益兑现后要立刻转向 latency/occupancy。

<sub>`vectorized_load` `paged_gather` `memory_bound` `latency_bound` `uncoalesced_access`  session `6270b3b2` · commit `14ea770`</sub>

### ✅ v2v3 launch overhead

工况: B300/sm103,CuTeDSL paged decode attention,bf16、hd256、GQA、q_per_seq=4,31 shape。
方法: V2 去掉 wrapper 里每次调用的 cu_seqlens_q.tolist() GPU->CPU 同步,换成 sync-free 的 q_per_seq=T//B(对 31 shape 的均匀连续前缀和精确成立,device 侧仍读 per-seq qs/klen,不需要 host 标量),@cute.kernel body/MMA/softmax/mask/grid 与 V1 逐字节不变;这一改也让函数可被 cudagraph 捕获。V3 加 _LAUNCH_CACHE(按 data_ptr 键),热路径跳过 ~112us 的 from_dlpack x7 + dim_order + mark_compact + empty_like 重建,只调 cached compiled executor(~11us)。
预期: 填掉 V1 测到的 ~141us/call host gap(ncu GPU 81.8us vs do_bench wall 223us),并解锁 cudagraph。
实测: V2 eager 196->100.7us(1.95x),cudagraph 首次可用 mean 67.0us(shape6 89.3、shape18 27.2);V3 eager 66.6us == cudagraph 67.1us;rel_err 0.00213,31/31。
归因与结论: 正向。.tolist() 既造成每次 device 同步、又直接阻断 cudagraph capture('Cannot copy between CPU and CUDA during capture');from_dlpack/mark 每次重建 ~112us 是 CuTeDSL launch 的隐藏大头 —— 按 data_ptr 缓存后 eager wall 收敛到 GPU 时间。结论: 消除任何 per-call GPU->CPU 同步对 CuTeDSL decode 是强制项(cudagraph 正确性 + 省 host);缓存 prepared launch 后,到 trtllm 的差距变成纯 GPU-kernel 问题。

<sub>`launch_optimization` `launch_overhead` `latency_bound`  session `6270b3b2` · commit `17b7f90`</sub>

### ✅ v4 cpasync pipeline

工况: B300/sm103,CuTeDSL paged decode attention,bf16、hd256、GQA、block_n=64、block_size {64,256},31 shape。
方法: 把同步 128-bit KV gather 改成 cp.async 2-stage 双缓冲软件流水 —— gather atom 换 CopyG2SOp(GLOBAL, 128-bit);sK/sV 用 N_STAGES=2 的 staged swizzled 布局(把 stage 编码成一个运行时可索引的 tensor MODE,sK[None,None,stage]);prologue 载 tile0 + commit,循环内预取 tile n+1、cp_async_wait_group(1) 让预取在飞、drain 当前 group、barrier;MMA/softmax/mask/grid 与 V3 不变。
预期: 打掉 long_scoreboard(3.36,主导 stall)—— 让下一 KV tile 的 HBM 读与当前 MMA 重叠、MLP 1->2,把有效带宽抬离 543 GB/s(7.5% SOL)地板,page256 的 16-tile shape 收益最大。
实测: eager 66.6->47.1us(1.41x),cudagraph 67.1->47.4us(1.42x);shape6 89.6->59.8(1.50x),BW 543->1192 GB/s;shapes 9/30 ->2030 GB/s(25-28% SOL);tflops 71.8;无 shape 回退;rel 0.00213,31/31;到 trtllm 差距 3.45x->2.44x。
归因与结论: 正向。双缓冲把 MLP 从 1 提到 2 tile 在飞,把 KV load 延迟藏进 Tensor-Core 工作后面,兑现最大收益恰在多-tile page256 regime。约束: 3-stage(224KB)超 SMEM optin(232448 B),只能 2-stage(160KB)。残留瓶颈: 小 grid(waves/SM 0.22,B*nkv << 148 SMs,小/单-tile shape 只有 1.9-2.7% SOL)+ 寄存器/TMEM 天花板(254 regs、寄存器驻留 O[M,256]、无 tcgen05/TMEM)—— 分别指向 V5 的 split-KV 与后续的 tcgen05 重写。

<sub>`async_copy` `multi_buffering` `overlap_pipelining` `latency_bound` `smem_lds_capacity`  session `6270b3b2` · commit `2346ae4`</sub>

### ✅ v9 tcgen05 qk tmem softmax

工况: B300/sm103,CuTeDSL(cutlass 4.5.2),单-tile M=64 N=64 D=256 的 QK^T decode shape,bf16 in、fp32 累加,单 CTA。
方法: 向量化 128-bit cp.async 进 swizzled tcgen05 SMEM -> tcgen05.MmaF16BF16Op(64,64,16) 做 QK^T -> 结果在 TMEM -> Ld16x64bOp readback 到寄存器 -> *scale_log2 -> exp2。
预期: 证明 tcgen05 QK^T + TMEM readback + 寄存器侧 softmax numerator 在 SM103 正确,作为 in-kernel 全融合的基座。
实测: vectorized cp.async+MMA 8.8us(比 scalar 28.5us 快 3.2x,de317cd);K=256 full QK^T shape 11.2us(e263755);QK+scale step1 10.9us(37b3e95);+exp2 step2 11.3us PASS(ac54f15)。
归因与结论: 正向。证明 tcgen05 QK^T + TMEM readback + 逐元素寄存器算术(scale、exp2)在 SM103 可行 —— 寄存器 readback 后做 softmax 数值运算 OK。仍缺: softmax 行归约(max/sum/normalize)、P@V、paged gather、多-tile online-softmax 循环。计量口径提醒: 11.3us 是单 CTA tiny tile 的 launch 主导值,不能和 trtllm(19.6us,31 shape,D256,多 CTA)比。这是通往 in-kernel 融合(V10)的最后一块可复用基座。

<sub>`tcgen05_umma` `tmem` `vectorized_load` `online_softmax` `latency_bound` `launch_overhead`  session `6270b3b2` · commit `ac54f15`</sub>

### ➖ cutedsl-v5 splitkv-cpasync

工况: B300/sm103,CuTeDSL(cutlass 4.5.2)实现的 paged decode attention,bf16 in/out、fp32 softmax/累加,hd256、GQA nqh=16、q_per_seq=4、block_n=64、block_size 64|256,用于 decode。
方法: V5 = split-KV + combine 归约 + cp.async 双缓冲(2-stage)KV 流水线;两个 GEMM 都用 warp-MMA(mma.sync 寄存器路径,非 tcgen05/TMEM),累加器常驻寄存器 fp32;KV 走 128-bit cp.async 的 pack-GQA paged gather。
预期: 建立正确且可精确复现的 CuTeDSL 性能锚点,为后续 tcgen05 融合(V9/V10)提供对照。
实测: eager 45.1µs / cudagraph 45.2µs,31/31 全部正确,max rel_l2=0.00213,与北园记录 45.1/45.3µs 精确复现。相对 vllm FA varlen(127.6µs)约 2.8× 快,相对 trtllm-gen 记录 19.3µs 仍有差距。
归因与结论: 中性/正向锚点。关键设计不变量: BLOCK_N=64 配合 block_size∈{64,256} 让每个 64-key tile 落在单一物理块内,从而 gather 可做单次合并的 128-bit cp.async;hd256 被 tiled-mma 拆成 16 的 k-chunk 用 warp-MMA 承载。教训: 该版是 memory-bound 形态、warp-MMA 未吃满 Tensor Core,是后续 tcgen05 in-kernel 融合的起点;复现基线时 rel_l2 与延迟需与记录逐位对齐以确认环境可信。

<sub>`split_kv` `async_copy` `multi_buffering` `paged_gather` `mma_scheduling` `online_softmax` `memory_bound` `latency_bound`  session `6270b3b2` · commit `9a14c6d`</sub>

### ➖ cudagraph gpu only metric

工况: B300/sm103,CuTeDSL decode kernel 与 flashinfer trtllm-gen baseline,bf16、hd256、GQA、q_per_seq=4,31 shape。
方法: 计量方法论 —— 判断 kernel 质量、复现 baseline 都用 cudagraph/GPU-only 口径而非 eager do_bench wall。
预期: 避免被 eager wall-clock 误导。
实测: eager wall 被 host-launch 主导 —— from_dlpack x7 + dim_order + mark_compact ~112us/call + ~60us compiled dispatch;每个 .item()/.tolist() 再加 ~50-60us 同步。这并非不可约的框架地板: 按 data_ptr 缓存 prepared launch 后 eager wall == cudagraph/GPU-only(V2 曾误判 'eager 打不过 trtllm 精简 C++ launch',是错的)。trtllm eager 恒 ~55-68us(被 seqused_k.max().item() 主机同步盖住、不随 shape 变),而 cudagraph GPU-only = 19.6us mean(随 KV 缩放)与记录 19.3us 吻合;CuTeDSL V5 eager 45.1 约等 cudagraph 45.2。
归因与结论: 中性/计量纪律。小 decode shape 上 launch/host 路径主导 wall-clock,只有 cudagraph/GPU-only 时间才反映 kernel 本身(也与 vLLM 用 cudagraph 捕获 decode 的生产口径一致)。可迁移规则: (1) 判断 CuTeDSL decode kernel 质量看 cudagraph/GPU-only,不看 eager do_bench wall;(2) 消除每次调用的 GPU->CPU 同步(cudagraph 捕获正确性 + 省 ~百 us host);(3) 复现 baseline 要用同一 GPU-only 镜头 —— trtllm 的 19.3us 只有在 GPU-only 下才能复现,eager 下会误看成 ~60us。

<sub>`launch_optimization` `launch_overhead`  session `6270b3b2` · commit `17b7f90`</sub>

### ➖ hd256 tmem o budget

工况: B300/sm103,CuTeDSL tcgen05 in-kernel FMHA,单 tile M=128 N=64 D=256(真实 head_dim),bf16、fp32 累加;TMEM 为 128 lane x 512 fp32 col。
方法: 从 D=64 扩到真实 D=256 单 tile —— 4-chunk QK + 在 PV 前归一化 P(P 只 64/行,寄存器放得下)+ 4-D-chunk PV + O(256) readback。
预期: 证明真实 hd256 链路正确,并刻画 D=256 的 TMEM/寄存器预算。
实测: Stage 3a rel_l2=0.0018 PASS,21.5us。关键约束: O 有 256 值/行 -> 放不进寄存器(超 254 reg);单独 O 的 TMEM accumulator 就占 512 列的 256 列(半个 TMEM)。
归因与结论: 中性约束,塑造后续设计。规则: hd256 下 O accumulator 独占 256 TMEM 列,S(64)+O(256)=320<=512 只能 single-buffer;寄存器驻留 O 的技巧(多-tile 用的 O_acc=O_acc*corr+O_t)在 D=256 不成立。可迁移做法: 单 tile 可在 PV 前归一化 P(只 64/行)避开 TMEM-O rescale、O 直接写出;但多-tile online-softmax 在 D=256 必须在 TMEM 里逐 D-chunk rescale O(Ld16x64/St16x64 读 O -> *corr -> 写回)。这条 TMEM 预算是 hd256 decode 的核心结构约束,决定 tile 尺寸与 buffering 决策。

<sub>`tmem` `tiling` `smem_lds_capacity` `register_pressure`  session `6270b3b2`</sub>

### ➖ late fullshape mean bf16 disproves combine wall

工况: B300/sm103,CuTeDSL a4 splitKV warp-spec decode,全 31 shape(hd256,klen 256×15/1024×16),per-shape 最优 NS,cudagraph GPU-only 口径。
方法: 先跑全 31 shape 正确性+性能坐实真实 mean(此前只在 shape0 报过 18.5/43µs);再把 partial 从 fp32 改 bf16(combine 读带宽减半),复测 mean 并对 mean-killer shape9(klen1024 B56)做 decode/combine 分解。
预期: 把 18.5µs 校正为全貌;bf16 应大幅削 klen=1024 大 B 的'combine partial 带宽墙'。
实测: a4 fp32 mean 47.71µs(min16.4/max116.3);bf16 mean 44.11µs(min15.55/max104.84),31/31 PASS(rel 0.0021→0.0029,bf16 精度损失可忽略)。四方对比:a4 44.1 / V5 45.3(max61.5)/ stage4 149.6(max398.9)/ trtllm ~19.3µs。bf16 只把 klen1024 B56 从 116 降到 104µs(~11%);分解 shape9 = decode 90.97µs + combine 19.89µs(decode 占 87%)。
归因与结论: 中性(校正+反超)。'18.5µs'只是 klen=256 小 B 特例,不泛化;真实 mean 44µs 已微超北园最好自研 V5、比原基线 stage4 快 3.4×。bf16 实测证伪了'combine 带宽墙'假设——klen=1024 大 B 的真瓶颈是 decode(87%)而非 combine;bf16 仍值得保留(免费小收益 + partial 内存减半)。教训: 报单-shape 峰值数会严重误导,必须用全 shape mean/max 坐实;优化前先做 decode/combine 分解定位真瓶颈,别照直觉猜'带宽墙',一个便宜的 bf16 实验就能证伪或证实假设。

<sub>`quantization` `split_kv` `memory_bound` `latency_bound`  session `3e24042c` · commit `f68fb22`</sub>

### ➖ late ncu trtllm 16warp vs 4warp

工况: B300/sm103,拿自研 tcgen05 paged decode(43µs full,decode 39µs)对标 flashinfer trtllm-gen cubin,shape0 klen=256 B=8 nqh=32。
方法: ncu 抓 trtllm 的注意力 kernel(--kernel-name 正则过滤掉 randn/mul/arange 等输入生成核)与自研 decode,对比 grid CTA 数、waves/SM、block 线程数、cluster、寄存器、KV tile、kernel 时长。
预期: 用实测结构定位差距根因,而非猜'占用率/cluster'。
实测: trtllm kernel = fmhaSm100 H256 PagedKv Causal P64 Q16 Kv128 Persistent;本机 GPU7 全路径 cudagraph 12.3µs(fmha 本体 9.3µs)。旧记录 19.6µs 是被占满的 GPU5 上测的。关键对比:trtllm 与自研 decode 同为 64 CTA、waves/SM=0.43、均不用 cluster/2-CTA MMA;唯一差别是 trtllm 每 CTA 512 线程(16 warp)vs 自研 128 线程(4 warp)。自研 decode 39µs vs trtllm 9.3µs。
归因与结论: 中性/诊断,但决定性重定向。同占用率下 16 warp 把 MMA/TMEM 异步延迟重叠藏住,4 warp(warp0 串行 QK→softmax→PV,零重叠)藏不住,这才是 ~4× 差距的根因。方向从'提 occupancy / 上 cluster / 2-CTA MMA'彻底转为'每 CTA 更多 warp + warp-specialize 重叠'。教训: 定位小-workload decode 瓶颈先 ncu 对标同类快核的 waves/线程数/cluster;当占用率、CTA 数、cluster 都相同而时长差数倍时,凶手往往是 per-CTA 的 warp 数(藏延迟能力),不是占用率。

<sub>`warp_specialization` `mma_scheduling` `latency_bound` `pipeline_stall` `occupancy_limited`  session `3e24042c`</sub>

### ➖ v0 baseline

工况: B300/sm103,CuTeDSL(cutlass 4.5.2)的 paged decode attention,bf16 in/out、fp32 softmax/累加,hd256、GQA nqh in {8,16,32}、nkv=2、q_per_seq=4、block_size {64,256},31 个 production shape,decode 阶段。
方法: V0 正确性基线 —— grid (B,nkv);qn*rep pack 进 M-tile;逐元素 scalar paged gather K/V 进 SMEM;warp MmaF16BF16Op(16,8,16) fp32 累加做 QK^T 与 P@V;online exp2 softmax(mn-view + quad-reduce);causal-tail mask;smem->gmem。
预期: 先把正确、能过 31 shape 的 Tensor-Core decode kernel立起来,给 Stage-2 优化当锚点。
实测: 延迟 mean 最高 2399.4us(shape9),shape6 2394.5us、shape0 696.6us、shape18 701.8us;tflops 3.133、tc_util 0.139%、bw 50.48 GB/s、bw_util 0.701%(0.08-0.70% SOL);rel_err 0.002126,31/31 PASS。
归因与结论: 延迟随 batch 几乎不变 → 瓶颈是 per-CTA 逐元素 scalar SMEM gather(uncoalesced),不是显存带宽;memory_bound、long_scoreboard 主导。可行的 Stage-2 攻击顺序(已记入 v0 pitfall):(1)向量化 128-bit cp.async paged gather,(2)双/三缓冲 KV 流水,(3)小 batch 用 split-KV,(4)tcgen05/TMEM 去掉寄存器驻留 O[M,256] 天花板。设计教训: M 要 pad 到 MMA 粒度并 mask padding 行;运行时用 cudaDevAttrMaxSharedMemoryPerBlockOptin(232448 B)确认 SMEM,因为 gpu-wiki 无 B300 的 SMEM/SM 数值。

<sub>`paged_gather` `mma_scheduling` `online_softmax` `memory_bound` `uncoalesced_access`  session `6270b3b2` · commit `0b7511f`</sub>

### ➖ v7 tcgen05 ptx finding

工况: B300/sm103,对比 CuTeDSL V5(45us,warp-MMA)与 Triton decode kernel(26us),bf16、hd256、GQA。
方法: 反汇编 26us Triton kernel 的 PTX(SM103),定位它比 V5 快 1.7x 的机制(纯分析,无代码改动)。
预期: 找到 Triton 优势来源,给 CuTeDSL 指方向。
实测: 关键发现 —— Triton 已经在发 tcgen05.mma + TMEM:tcgen05.alloc 512 cols、QK^T 4 次 tcgen05.mma、P@V 16 次、tcgen05.ld/st.16x32bx2 做 TMEM<->reg 的 softmax readback/P-O writeback、cp.async KV、mbarrier pipeline。CuTeDSL V5 用 legacy warp.MmaF16BF16Op(mma.sync + 寄存器累加),这正是 1.7x 差距。研究性结论,无新 kernel 数字。
归因与结论: 中性方向性结论。速度差是硬件 path 差异(tcgen05/TMEM vs mma.sync 寄存器累加),而非调参可补 —— 寄存器累加把 O[M,256] 钉在寄存器里(254 regs)是结构性天花板。可迁移规则: 当一个自动编译器(Triton)在同硬件上明显更快时,先看它的 PTX 用了哪些指令 path;这里的修法是让 CuTeDSL 显式换到 tcgen05.MmaF16BF16Op + TmemAllocator + make_smem_layout_a/b,复刻 Triton 自动命中的 tcgen05 path。

<sub>`tcgen05_umma` `tmem` `mma_scheduling` `register_pressure` `instruction_issue_bound`  session `6270b3b2` · commit `3bfb22d`</sub>

### ❌ constexpr unroll llvm wall

工况: B300/sm103(sm_103a lowering),CuTeDSL(4.5.2),把全部组件组装成单 CTA 处理全 klen 的 decode kernel(D=256,最多 4 KV tile)。
方法: 组装完整分页 kernel(GQA pack + paged gather + V 转置 + causal-tail + 多-tile online softmax + D=256 TMEM-O rescale);编译失败后用 num_tiles 强制 1 做 bisect。
预期: 让集成 kernel 编译通过并对 reference.py 验证。
实测: 单-CTA all-klen kernel 过了 MLIR 层,但报 'Failed creating llvm::Module / serializing the module'(sm_103a)。bisect 决定性证据: 强制 num_tiles=1 -> 编译通过并运行(其余逻辑全 OK)-> 根因 = KV 循环 constexpr 全展开(4 tile x tcgen05 MMA + TMEM 操作)超 LLVM module 上限。尝试改动态 cutlass.range 让 body 只编一次 -> 撞 'src is structured different after this for'(loop-carried run_max/run_sum 初始是编译期常量、循环内变动态,结构不一致;+0 被常量折叠;把 rMax/rSum 张量化也没满足 loop-carry 语法)。
归因与结论: 负向/过程教训。规则: CuTeDSL constexpr 循环会把 IR 全展开,重-tcgen05 body x 多 tile 会撑爆 LLVM module 序列化 —— 症状是 MLIR 通过但 LLVM 创建失败,用 num_tiles=1 一测即可定位。正解: split-KV(每 CTA 只处理 1-2 tile,constexpr 展开自然变小 + grid 从 B*nkv 扩到 B*nkv*num_splits 提 occupancy),combine 复用 V5 的 —— split-KV 在此不仅是性能优化,更是让集成 kernel 能编译的前提。备选是掌握 cutlass.range 正确的 loop-carried-values 语法(携带 online-softmax 状态),本次未掌握。

<sub>`split_kv` `tmem` `x-compiler_ir_size`  session `6270b3b2`</sub>

### ❌ cutedsl compile pitfalls

工况: B300/sm103,CuTeDSL(cutlass 4.5.2),在双缓冲流水(V4)与 tcgen05 集成(Stage 3c)反复踩到的编译期坑。
方法: 把这些坑及修法汇编成可复用清单(减少多轮分钟级慢编译)。
预期: 沉淀成清单,后续直接照做。
实测(条件性,无性能数): (1) @cute.kernel body 不 close over @cute.jit host 局部 -> 编译期值必须作 Constexpr/typed kernel 参数(否则 'range_constexpr requires constexpr' 或值变运行时);(2) range_constexpr 用语句形式做编译期循环、不放 comprehension;(3) 双缓冲 stage 编码成运行时 tensor MODE(staged layout,索引 [...,stage]),不要用 per-stage Python list;(4) swizzled sQ/sV 不能 scalar/plain 写 -> 先写 plain sQn/sVn 再 relayout(autovec_copy)进 swizzled MMA tensor;(5) coord/identity tensor 不能 make_tensor(coord iterator) 展平 -> 直接线性索引 tScS[i];(6) 捕获 tensor 的闭包在动态控制流里不支持 -> 内联表达式;(7) make_tmem_copy 要作用在 flat_divide 的 2D epi-tile 切片(tCtS[(None,None),0,0]),且需干净 128 行 fragment(make_trivial_tiled_mma,M=64 复合 (16,4) 不适配);(8) autovec_copy 要求源/目的位宽相等(fp32->bf16 先 .to(bf16)),16-bit cp.async 不支持(只 32/64/128)。
归因与结论: 负向/过程。每条都是 CuTeDSL 的 layout/legalization 约束。可迁移主线: 编译期值一律走 Constexpr 参数;swizzled tensor 经 plain 暂存缓冲 + relayout 处理;TMEM copy 只作用在干净 2D 切片。这些坑单看细碎,但每个都是一轮慢编译的代价,提前照清单写能显著提速 tcgen05 kernel 的落地。

<sub>`tmem` `multi_buffering` `swizzle` `x-dsl_compile`  session `6270b3b2`</sub>

### ❌ late dynamic range tmem copy limit

工况: B300/sm103,CuTeDSL(cutlass 4.5.2),方案 C(单 kernel in-kernel online-softmax 去掉 combine)。需一个 CTA 处理 klen=1024=16 KV tile,故需循环;循环内要用 tcgen05 的 make_tmem_copy 读写 TMEM 中的 S/P/O。
方法: 用 cutlass.range(unroll=1)(不展开,body 只编译一次)做动态 KV 循环,携带 loop-carried online-softmax 状态(run_max/run_sum 标量 + O_acc 寄存器 in-place)。
预期: 动态循环避免 16-tile constexpr 全展开导致 LLVM IR 爆(stage3c 在 4-tile D=256 已爆过)。
实测: 编译全失败,报 MLIRError / ICE。逐一试过 atom 建在循环内/外、内联 get_slice(thr-copy)、经函数(softmax_step)边界、循环边界改成运行时 Int32 参数——都不行。最小 repro 也确认:make_tmem_copy 用于动态 cutlass.range 循环内(无论直接/经函数)都无法 legalize(builtin.unrealized_conversion_cast: atom.tmem_load→tiled_copy remained live)。两 python 都是 cutlass 4.5.2、FA4 能跑动态循环 → 是写法结构问题不是版本(FA4 靠全 warp-specialize 重机器绕过)。绕过办法:range_constexpr 按 num_tiles 特化编译(生产 klen 只有 4/16 两值),已验证 v3 D=64 16-tile constexpr 编译+PASS(rel=0.0013),后续 a3 D=256 16-tile constexpr 编译约几分钟也 PASS 不爆 IR。
归因与结论: 负向(硬 DSL 约束)。cutlass 4.5.2 上 tcgen05 tmem-copy atom 不能跨动态 scf.for region legalize。教训: 简单 128 线程核里别对含 TMEM-copy 的循环用动态 cutlass.range;CuTeDSL 本就按整数常量特化,直接用 range_constexpr 按有限的 num_tiles 值特化编译,精简 per-tile body 后 16-tile 也不爆 IR。此坑与 warp-specialize 的 scf.if 同源(atom 不能跨 scf region 边界)。

<sub>`tmem` `in_kernel_fusion` `x-constexpr-specialization` `x-compiler-legalization`  session `3e24042c`</sub>

### ❌ late latency bound tmem occupancy wall

工况: B300/sm103,CuTeDSL tcgen05 分页 decode,mean-killer shape9 klen1024 B56 NS=8(~89.8µs),hd256/D=256/M_TILE=128,5 warp/1 CTA per SM。
方法: ncu 在真实工作点剖 stall reasons(不用占用率不足的 NS=1 假象点),再系统实测各降延迟/减 work/提占用杠杆:within-CTA S 双缓冲+QK 预取(2a)、tps=1 去 rescale、向量化 Q-load+O-write、env-gated skip 隔离各段成本。
预期: 找到能把 klen1024 大 B decode 压向 V5 61µs 的可行杠杆。
实测: waves/SM=3.03(占用率够,非 CTA 不足)、sm_throughput 7.28%、warps_active 仅 7.78%(trtllm 21%,~3× headroom);long_scoreboard(访存/TMEM 读延迟)占 61% stall cycles。杠杆结果:within-CTA pipeline(2a)−10%;tps=1(去全部 rescale)−20%(per-CTA 固定开销翻倍抵消);向量化 Q/O ≈0%(256 标量本已重叠成一次延迟等待);skip Q-gather 省 3.4µs、skip O-write 省 8.6µs、两者都 skip 省 12µs(13%)。
归因与结论: 负向(架构上限图)。decode 是延迟受限——compute warp 读 TMEM(S 读 + O rescale 读写 + O readback)延迟暴露,只有 5 warp/1 CTA per SM,warp 等待时无别的 warp 顶上。三条藏延迟路被硬资源堵死:(1)2 CTA/SM 不可行——不只是 smem(sQ64+sK+sV=128KB>116KB),更是 TMEM:单 CTA 的 O 累加器就占 512 列中的 256,两 CTA 的 O(512 列)放不下,故 M=64 减 smem 也救不了;(2)批量加载 4 个 O D-chunk 藏延迟 → 256 reg/线程(现 108)寄存器爆;(3)向量化对延迟受限无效(只减指令数不减延迟)。教训: latency-bound + 1 CTA/SM 时,减 work/向量化/去 rescale 都无效,只有'更多并发 warp'能藏延迟;判占用率上限要同时看 smem 和 TMEM 两个约束——D=256 时 O 独占半个 TMEM,是比 smem 更硬的 2-CTA/SM 天花板。

<sub>`tmem` `register_blocking` `occupancy_tuning` `latency_bound` `occupancy_limited` `register_pressure` `smem_lds_capacity` `pipeline_stall`  session `3e24042c`</sub>

### ❌ late tma tensor descriptor stall

工况: B300/sm103,CuTeDSL(cutlass 4.5.2)。为把 decode 重写成 trtllm 级(TMA + persistent + lean pipeline)先给最高风险项 TMA 去风险,以 FA4 flash_fwd_sm100 为参考搭独立 TMA paged-KV tile 载入 PoC。
方法: 从零写 ~30 个 PoC 变体,逐一匹配 FA4:plain 与 swizzled smem layout、make_tiled_tma_atom_B、quack 的 tma_get_copy_fn、2D 与 4D gmem 张量、PipelineTmaAsync wrapper、cluster launch、assume_tensor_aligned 的 stride 对齐标记、D=128/256、system python 与 xingze conda 两个 env;并逐一隔离 copy vs wait vs expect_tx。
预期: 确认独立 TMA tile 载入在本环境可行,再建完整流水。
实测: 每个 tensor-TMA copy 变体都 device-hang——连 fire-and-forget(无 wait)也挂,说明 copy 指令本身 device-fault,不是 tx/wait。关键隔离:expect_tx 不带 copy 能干净完成;FA4 自身 forward 在 xingze env PASS(CC 10.3,0.045ms)证明 TMA 硬件可用;简单 cp.async.bulk(非 tensor、无 descriptor)standalone 成功(max_err=0)。compute-sanitizer 无法用(xingze env CUDA runtime 13.3 vs driver 13.2 不匹配无法 attach;system env 无法报告 hung kernel)。
归因与结论: 负向(工具链/DSL 上限)。卡点精确定位在 tensor-descriptor 路径(make_tiled_tma_atom_B),不是 async-bulk 机制、mbarrier/expect_tx/wait 或 env;从零的 tensor-TMA 只在 FA4 完整 persistent/warp-spec kernel 上下文里才工作,piecemeal 无法复现。重要修正认知:简单 cp.async.bulk 载入可用 → load 从来不是真瓶颈(简单 bulk-copy re-paged-128 的连续 KV 页即可),tensor-TMA 并非必需;真瓶颈是 compute-warp 结构(M=128 多为 padding + 4 warp 藏不住 TMEM 读延迟)。教训: 别从零死磕 tensor-TMA descriptor——先用 cp.async.bulk 验证 async-bulk 通路,tensor-TMA 若要用应直接改用/复用完整可跑参考核而非最小 PoC 逐位拼;device-hang 且 fire-and-forget 也挂=copy 指令本身 fault。

<sub>`async_copy` `x-tma-descriptor` `x-toolchain-limit` `latency_bound`  session `3e24042c`</sub>

### ❌ late within cta overlap serial o chain

工况: B300/sm103,CuTeDSL warp-spec 分页 decode(a4=44µs 基座),shape0 klen256 B8、mean-killer shape9/30 klen1024 B56,hd256/GQA。
方法: 为冲 mean<30µs 写两个完整新核。a6=FA4 式 10-warp 流水:专用 mma(warp8)/softmax(0-3)/correction(4-7)/load(9)warp,S/P 在 TMEM 双缓冲,correction 因子经 sScale smem 广播。a7=N=128(Kv128)把 KV tile 从 64 增到 128,减半 tile 数(tps=1 时一个 CTA 一 tile,免 rescale)。
预期: a6 用 within-CTA warp 重叠藏 MMA/TMEM 延迟;a7 用更大 tile 减少 latency-bound 的 per-tile work(S-read/rescale/readback)。
实测: 两核均正确但均不更快。a6 shape9 decode 100µs vs a4 91µs(~+10%);a7 shape9 full 105 vs 104µs、shape0 22.4 vs 17.3µs(略慢)。全部 rel PASS。
归因与结论: 负向。a6 慢的根因:O 累加器是串行依赖链(每 tile 的 PV 需上一 tile 的 O 已 rescale),加 warp/流水级数无法缩短这条链,只增加 barrier 开销;within-CTA 重叠对这条串行链无效。a7 慢的根因:kernel 是 latency-bound,减少 work 不减少未藏住的延迟,且 N=128 带来更大 MMA + 2-half PV + 2-block gather 抵消了减半 tile 的收益。两个可迁移坑:(a6)elect_one 放在 if warp_idx<SOFTMAX_WARPS(4 warp)内 → 每 warp 各 arrive 一次 = 对 count-1 barrier arrive 4 次,parity 翻回原值,mma 的 wait 永不满足 → 死锁,需恰好一次 arrive;(a7)block_size=64 但 N=128 时一个 tile 跨两个物理页,须从各自 block_table 项分别 gather 每个 64-key 子块,否则 keys 64-127 读到相邻垃圾内存(症状:QK/run_max 就错,P/V 改动不影响结果)。教训: 对'输出累加器有串行依赖'的 kernel,within-CTA 多 warp/流水重叠不能加速,只加 barrier 税;latency-bound 下'减少 work'的招普遍无效;跨页的大 KV tile 必须按物理页边界分段 gather。

<sub>`warp_specialization` `tcgen05_umma` `tmem` `overlap_pipelining` `online_softmax` `latency_bound` `pipeline_stall` `barrier_sync`  session `3e24042c`</sub>

### ❌ v6 tilesize tuning fail

工况: B300/sm103,CuTeDSL paged decode attention(V5 split-KV,45.1us),bf16、hd256、GQA、block_size {64,256}。
方法: tile-size tuning —— 试 block_n=32 与 block_n=128,目标是把 occupancy 抬离 254-reg / 6.2%-occupancy 墙。
预期: 在大 non-split page256 shape(距 trtllm 2.33x)上突破 warp-MMA occupancy 天花板。
实测: FAIL,无提升,回退 V5。block_n=32 回退(46.2 > 45.1us,tile 数翻倍 + per-tile 开销盖过 occupancy 收益,大 shape 60->75us);block_n=128 直接 CUDA_ERROR_INVALID_VALUE(2x128x256x2B/stage 超 SMEM optin)。
归因与结论: 负向但极有价值。根因: 大 shape 上 grid=64 < 148 SMs,occupancy 是 GRID-limited,不是 SMEM/reg-limited —— 只有 64 个 block 时,每 SM 塞第二个 block 物理上不可能,所以任何 tile 调参都动不了 occupancy。可迁移规则: 当 grid_CTAs < num_SMs 时,先算 grid/SM,占用率由 grid 决定,tile/stage/reg 调参无效;要么 split-KV 增加 CTA 数,要么换结构性的 MMA path(tcgen05/UMMA+TMEM 把寄存器驻留 O[M,256] 移出寄存器)。结论: V5 是 warp-MMA 架构的实际天花板,剩余 2.33x 只能靠 tcgen05 重写。

<sub>`tiling` `occupancy_limited` `grid_underutilization` `register_pressure` `smem_lds_capacity`  session `6270b3b2` · commit `8e5b080`</sub>

### ❌ v8 triton ceiling

工况: B300/sm103,Triton decode kernel + CuTeDSL tcgen05 基座验证,bf16、hd256、GQA nqh in {8,16,32}、q_per_seq=4,31 shape。
方法: (a) 对 Triton 做穷举 heuristic tuning(split sweep sp=1..4、num_warps、packed_base 门限);(b) 证明 CuTeDSL tcgen05 积木可用。
预期: 把 Triton 推到极限,同时给 CuTeDSL 重写验证 tcgen05 building blocks。
实测: Triton 触底 26.2-26.3us(shapes 16,17 各改善 3.8-5.9us;shape16 sp=4/nw=4=27.9 vs sp=3/nw=8=31.6),rel 0.00215,31/31。但大 rep=16 page256 shape(7-9,28-30)卡在 42-46us,254 regs、~6% occupancy、instruction_throughput_limited。tcgen05 基座证明: scalar SMEM write + tcgen05.MmaF16BF16Op(64,64,16) + TMEM accumulator + Ld16x64bOp readback 在 SM103 可编译可跑。
归因与结论: 负向(Triton 到顶)+ 正向基座证明。大 page256 shape 已到 Triton-on-SM103 天花板,BLOCK_N/stages/warps/splits 全都动不了 —— 差距需要手调 tcgen05 scheduling/TMA,Triton 能自动命中 tcgen05 指令但达不到手写 CUDA 的水平。规则: 自动编译器的调参空间有其硬上限,越过它必须换手写 tcgen05 path;好在同一批 tcgen05 积木(MMA(64,64,16)+TMEM accumulator+Ld16x64bOp readback)已在 SM103 落地,为 V9/V10 的 CuTeDSL 重写铺好路。

<sub>`occupancy_tuning` `tcgen05_umma` `tmem` `instruction_issue_bound` `occupancy_limited` `register_pressure`  session `6270b3b2`</sub>

