# Paged Attention Decode on sm103 — Pitfalls

Traps hit while optimizing `paged_attention_decode` — CuTeDSL (cutlass 4.5.2), bf16 in/out, fp32 partial/accum, hd256 (D=256), GQA, split-KV num_splits, combine 归约 on B300 / Blackwell / sm103.
Companion to:

- Optimization journey: [journey](../../../../ref-docs/nvidia/cutedsl/sm103/paged_attention_decode-optimization.md)
- Optimization highlights: [highlights](../../../../kernel-opt/nvidia/cutedsl/sm103/paged_attention_decode.md)

---

## 1. constexpr unroll llvm wall

**Trap**: 把完整分页 decode kernel(GQA pack + paged gather + V 转置 + causal-tail + 多-tile online softmax + D=256 TMEM-O rescale)组装成单 CTA 处理全部 KV tile;编译失败后用 num_tiles 强制成 1 做 bisect（预期 让集成后的单-CTA kernel 编译通过并跑通 reference.py）

**Result**: 单-CTA all-klen kernel 过了 MLIR 但报 'Failed creating llvm::Module / serializing the module'(sm_103a)。bisect: 强制 num_tiles=1 编译通过并运行 -> 根因是 KV 循环 constexpr 全展开(4 tile x tcgen05 MMA + TMEM ops)超出 LLVM module 上限。改动态 cutlass.range 让 body 只编一次 -> 撞 'src is structured different after this for'(loop-carried run_max/run_sum 从 constexpr 变 dynamic 结构不一致;+0 被折叠;张量化 rMax/rSum 也没满足 loop-carry 语法)

**Why**: constexpr 循环会全展开 IR,4-tile x 重 tcgen05 body 撑爆 LLVM module。正解是 split-KV(每 CTA 只处理 1-2 tile -> IR 小 + occupancy 高),复用 V5 combine kernel —— split-KV 不只是性能,还是让集成 kernel 能编译的前提;或掌握 cutlass.range 正确的 loop-carried-values 语法(本次未掌握)

**Lesson**: 负向/过程教训。规则: CuTeDSL constexpr 循环会把 IR 全展开,重-tcgen05 body x 多 tile 会撑爆 LLVM module 序列化 —— 症状是 MLIR 通过但 LLVM 创建失败,用 num_tiles=1 一测即可定位。正解: split-KV(每 CTA 只处理 1-2 tile,constexpr 展开自然变小 + grid 从 B*nkv 扩到 B*nkv*num_splits 提 occupancy),combine 复用 V5 的 —— split-KV 在此不仅是性能优化,更是让集成 kernel 能编译的前提。备选是掌握 cutlass.range 正确的 loop-carried-values 语法(携带 online-softmax 状态),本次未掌握。

<sub>`split_kv` `tmem` `x-compiler_ir_size`  session `6270b3b2`</sub>

## 2. cutedsl compile pitfalls

**Trap**: 跨双缓冲流水(V4)与 tcgen05 集成(Stage 3c)反复踩到的 CuTeDSL 4.5.2 编译期坑及其修法(汇编成清单)（预期 把这些编译坑沉淀成可复用清单,减少后续多轮慢编译）

**Result**: 坑与修法: (1) @cute.kernel body 不 close over @cute.jit host 局部变量 -> 所有编译期值必须作 Constexpr/typed kernel 参数传(否则报 'range_constexpr requires constexpr' 或值变运行时);(2) range_constexpr 编译期循环用语句形式、绝不放进 comprehension;(3) 双缓冲 stage 编码成运行时 tensor MODE(staged layout,[...,stage])而非 per-stage Python list;(4) swizzled sQ/sV 不能 scalar/plain 写 -> 先写 plain sQn/sVn 再 relayout(autovec_copy)进 swizzled MMA tensor;(5) coord/identity tensor 不能用 make_tensor(coord iterator) 展平 -> 直接线性索引 tScS[i];(6) 捕获 tensor 的闭包在动态控制流里不支持 -> 内联;(7) make_tmem_copy 要作用在 flat_divide 的 2D epi-tile 切片(tCtS[(None,None),0,0])且需干净 128 行 fragment(make_trivial_tiled_mma);(8) autovec_copy 要求源/目的位宽相等(fp32->bf16 需先 .to(bf16)),16-bit cp.async 不支持(只 32/64/128)

**Why**: 每条都是 CuTeDSL 的 layout/legalization 约束;可迁移主线是 —— 编译期值一律走 Constexpr 参数、经 plain 暂存缓冲 + relayout 处理 swizzled tensor、TMEM copy 只作用在干净 2D 切片上

**Lesson**: 负向/过程。每条都是 CuTeDSL 的 layout/legalization 约束。可迁移主线: 编译期值一律走 Constexpr 参数;swizzled tensor 经 plain 暂存缓冲 + relayout 处理;TMEM copy 只作用在干净 2D 切片。这些坑单看细碎,但每个都是一轮慢编译的代价,提前照清单写能显著提速 tcgen05 kernel 的落地。

<sub>`tmem` `multi_buffering` `swizzle` `x-dsl_compile`  session `6270b3b2`</sub>

## 3. late dynamic range tmem copy limit

**Trap**: 方案 C 要一个 CTA 跑 16 tile 的动态 KV 循环:试 cutlass.range(unroll=1) 动态循环 + loop-carried online-softmax 状态,循环内用 tcgen05 make_tmem_copy（预期 动态(不展开)循环避免 16-tile constexpr 全展开导致 LLVM IR 爆(stage3c 在 4-tile D=256 就爆过)）

**Result**: 所有写法都 MLIRError/ICE:atom 建循环内/外、内联 get_slice、经函数边界、运行时 vs 常量循环边界——全部失败(builtin.unrealized_conversion_cast: atom.tmem_load→tiled_copy 在动态 scf.for region 内无法 resolve / remained live)。绕过:range_constexpr 特化,v3 D=64 16-tile constexpr 编译+PASS(rel=0.0013),后续 a3 D=256 16-tile constexpr 亦 PASS

**Why**: cutlass 4.5.2 里 make_tmem_copy 产生的 tmem-copy atom 无法在动态 scf 循环 region 内 legalize;FA4 能用动态循环是靠全 warp-specialize 的重机器。CuTeDSL 本就按 num_tiles(仅 4/16 两值)特化编译,constexpr 展开可绕过,且 warp-spec 精简 per-tile body 让 16-tile 不爆 IR

**Lesson**: 负向(硬 DSL 约束)。cutlass 4.5.2 上 tcgen05 tmem-copy atom 不能跨动态 scf.for region legalize。教训: 简单 128 线程核里别对含 TMEM-copy 的循环用动态 cutlass.range;CuTeDSL 本就按整数常量特化,直接用 range_constexpr 按有限的 num_tiles 值特化编译,精简 per-tile body 后 16-tile 也不爆 IR。此坑与 warp-specialize 的 scf.if 同源(atom 不能跨 scf region 边界)。

<sub>`tmem` `in_kernel_fusion` `x-constexpr-specialization` `x-compiler-legalization`  session `3e24042c`</sub>

## 4. late latency bound tmem occupancy wall

**Trap**: ncu 在真实工作点(shape9,NS=8,~89.8µs)剖 stall reasons,并系统测各'减 work / 提占用'杠杆(within-CTA pipeline、tps=1 去 rescale、向量化 Q/O、skip 隔离)（预期 找到能藏住/减少延迟的可行杠杆,把 klen1024 大 B 的 decode 压向 V5 的 61µs）

**Result**: waves/SM=3.03(占用够)、sm_throughput 7.28%、warps_active 7.78%(trtllm 21%);long_scoreboard 占 61% stall cycles。杠杆:within-CTA pipeline −10%;tps=1 −20%;向量化 Q/O ≈0%;skip Q-gather 省 3.4µs、skip O-write 省 8.6µs

**Why**: decode 延迟受限:compute warp 读 TMEM(S 读 + O rescale 读写 + O readback)延迟暴露,仅 5 warp/1 CTA per SM,warp 等待时无别的 warp 顶上。三条藏延迟路被硬资源堵死

**Lesson**: 负向(架构上限图)。decode 是延迟受限——compute warp 读 TMEM(S 读 + O rescale 读写 + O readback)延迟暴露,只有 5 warp/1 CTA per SM,warp 等待时无别的 warp 顶上。三条藏延迟路被硬资源堵死:(1)2 CTA/SM 不可行——不只是 smem(sQ64+sK+sV=128KB>116KB),更是 TMEM:单 CTA 的 O 累加器就占 512 列中的 256,两 CTA 的 O(512 列)放不下,故 M=64 减 smem 也救不了;(2)批量加载 4 个 O D-chunk 藏延迟 → 256 reg/线程(现 108)寄存器爆;(3)向量化对延迟受限无效(只减指令数不减延迟)。教训: latency-bound + 1 CTA/SM 时,减 work/向量化/去 rescale 都无效,只有'更多并发 warp'能藏延迟;判占用率上限要同时看 smem 和 TMEM 两个约束——D=256 时 O 独占半个 TMEM,是比 smem 更硬的 2-CTA/SM 天花板。

<sub>`tmem` `register_blocking` `occupancy_tuning` `latency_bound` `occupancy_limited` `register_pressure` `smem_lds_capacity` `pipeline_stall`  session `3e24042c`</sub>

## 5. late tma tensor descriptor stall

**Trap**: 为 trtllm 级重写去风险,从零搭 TMA paged-KV 载入 PoC(~30 变体:plain/swizzled layout、make_tiled_tma_atom_B、quack tma_get_copy_fn、4D gmem、PipelineTmaAsync wrapper、cluster launch、stride 对齐标记、D=128/256、system 与 xingze env)（预期 先确认独立 TMA tile 载入可行,再建完整 persistent/warp-spec 流水）

**Result**: 每个 tensor-TMA copy 变体都 device-hang(连 fire-and-forget/无 wait 都挂,即 copy 指令本身 device-fault);expect_tx 无 copy 能完成;FA4 forward 在 xingze env PASS(CC10.3,0.045ms)证明 TMA 硬件可用;简单 cp.async.bulk(非 tensor、无 descriptor)standalone 成功(max_err=0)

**Why**: 卡的是 tensor-descriptor 路径(make_tiled_tma_atom_B),不是 async-bulk 机制、mbarrier/expect_tx/wait 或 env;从零的 tensor-TMA 只在 FA4 完整 persistent/warp-spec kernel 上下文里才工作,无法 piecemeal 复现

**Lesson**: 负向(工具链/DSL 上限)。卡点精确定位在 tensor-descriptor 路径(make_tiled_tma_atom_B),不是 async-bulk 机制、mbarrier/expect_tx/wait 或 env;从零的 tensor-TMA 只在 FA4 完整 persistent/warp-spec kernel 上下文里才工作,piecemeal 无法复现。重要修正认知:简单 cp.async.bulk 载入可用 → load 从来不是真瓶颈(简单 bulk-copy re-paged-128 的连续 KV 页即可),tensor-TMA 并非必需;真瓶颈是 compute-warp 结构(M=128 多为 padding + 4 warp 藏不住 TMEM 读延迟)。教训: 别从零死磕 tensor-TMA descriptor——先用 cp.async.bulk 验证 async-bulk 通路,tensor-TMA 若要用应直接改用/复用完整可跑参考核而非最小 PoC 逐位拼;device-hang 且 fire-and-forget 也挂=copy 指令本身 fault。

<sub>`async_copy` `x-tma-descriptor` `x-toolchain-limit` `latency_bound`  session `3e24042c`</sub>

## 6. late within cta overlap serial o chain

**Trap**: 两个完整新核冲 mean<30µs:a6=FA4 式 10-warp 流水(专用 mma/softmax/correction/load warp、S/P TMEM 双缓冲、corr 经 sScale smem 广播);a7=N=128 Kv128(减半 tile 数)（预期 within-CTA warp 重叠(a6)藏住 MMA/TMEM 延迟;更大 tile(a7)减少 latency-bound work）

**Result**: 均正确但均不更快。a6 shape9 decode 100 vs a4 91µs(~+10%);a7 shape9 full 105 vs 104、shape0 22.4 vs 17.3µs(略慢)。全部 rel PASS

**Why**: O 累加器是串行依赖链(每 tile 的 PV 需上一 tile 的 O 已 rescale),加 warp/流水级数无法缩短它,只增 barrier 开销;且 kernel latency-bound,减少 work(N=128 更大 MMA + 2-half PV + 2-block gather)不减少未藏住的延迟

**Lesson**: 负向。a6 慢的根因:O 累加器是串行依赖链(每 tile 的 PV 需上一 tile 的 O 已 rescale),加 warp/流水级数无法缩短这条链,只增加 barrier 开销;within-CTA 重叠对这条串行链无效。a7 慢的根因:kernel 是 latency-bound,减少 work 不减少未藏住的延迟,且 N=128 带来更大 MMA + 2-half PV + 2-block gather 抵消了减半 tile 的收益。两个可迁移坑:(a6)elect_one 放在 if warp_idx<SOFTMAX_WARPS(4 warp)内 → 每 warp 各 arrive 一次 = 对 count-1 barrier arrive 4 次,parity 翻回原值,mma 的 wait 永不满足 → 死锁,需恰好一次 arrive;(a7)block_size=64 但 N=128 时一个 tile 跨两个物理页,须从各自 block_table 项分别 gather 每个 64-key 子块,否则 keys 64-127 读到相邻垃圾内存(症状:QK/run_max 就错,P/V 改动不影响结果)。教训: 对'输出累加器有串行依赖'的 kernel,within-CTA 多 warp/流水重叠不能加速,只加 barrier 税;latency-bound 下'减少 work'的招普遍无效;跨页的大 KV tile 必须按物理页边界分段 gather。

<sub>`warp_specialization` `tcgen05_umma` `tmem` `overlap_pipelining` `online_softmax` `latency_bound` `pipeline_stall` `barrier_sync`  session `3e24042c`</sub>

## 7. v6 tilesize tuning fail

**Trap**: 在 V5(split-KV,45.1us)基础上做 tile-size tuning: 试 block_n=32 和 block_n=128,想把 occupancy 抬离 254-reg / 6.2%-occupancy 的墙（预期 在大 non-split page256 shape(距 trtllm 2.33x)上突破 warp-MMA 的 occupancy 天花板）

**Result**: FAIL,无提升,回退 V5。block_n=32 回退(46.2 > 45.1us;tile 数翻倍 + per-tile 开销 > occupancy 收益,大 shape 60->75us);block_n=128 报 CUDA_ERROR_INVALID_VALUE(2x128x256x2B/stage 超 SMEM optin)

**Why**: 大 shape 上 grid=64 < 148 SMs,occupancy 是 GRID-limited 而非 SMEM/reg-limited —— 只有 64 个 block 时每 SM 第二个 block 根本排不下,所以调 tile 无法提 occupancy;剩余 2.33x 需要 tcgen05/UMMA+TMEM 的结构性 MMA-path 改动,不是参数能调出来的。V5 是 warp-MMA 架构的实际天花板

**Lesson**: 负向但极有价值。根因: 大 shape 上 grid=64 < 148 SMs,occupancy 是 GRID-limited,不是 SMEM/reg-limited —— 只有 64 个 block 时,每 SM 塞第二个 block 物理上不可能,所以任何 tile 调参都动不了 occupancy。可迁移规则: 当 grid_CTAs < num_SMs 时,先算 grid/SM,占用率由 grid 决定,tile/stage/reg 调参无效;要么 split-KV 增加 CTA 数,要么换结构性的 MMA path(tcgen05/UMMA+TMEM 把寄存器驻留 O[M,256] 移出寄存器)。结论: V5 是 warp-MMA 架构的实际天花板,剩余 2.33x 只能靠 tcgen05 重写。

<sub>`tiling` `occupancy_limited` `grid_underutilization` `register_pressure` `smem_lds_capacity`  session `6270b3b2` · commit `8e5b080`</sub>

## 8. v8 triton ceiling

**Trap**: 两条线: (a) 对 Triton kernel 做穷举 heuristic tuning(split sweep sp=1..4、num_warps、packed_base 门限);(b) 证明 CuTeDSL tcgen05 积木在 SM103 可编译可跑（预期 把 Triton 推到极限,同时验证 tcgen05 building blocks,为 CuTeDSL 重写铺路）

**Result**: Triton 触底 26.2-26.3us(shapes 16,17 各改善 3.8-5.9us;shape16 sp=4/nw=4=27.9 vs sp=3/nw=8=31.6us),rel 0.00215,31/31。但大 rep=16 page256 shape(7-9,28-30)卡在 42-46us,254 regs、~6% occupancy、instruction_throughput_limited。tcgen05 CuTeDSL 基座证明: scalar SMEM write + tcgen05.MmaF16BF16Op(64,64,16) + TMEM accumulator + Ld16x64bOp readback 在 SM103 可编译可跑

**Why**: 大 page256 shape 已到 Triton-on-SM103 天花板 —— 任何 BLOCK_N/stages/warps/splits 都动不了;到 trtllm(19.3us)的差距需要手调 tcgen05 scheduling/TMA(Triton 能自动命中但调不到手写 CUDA 的程度)。这坐实了必须转 CuTeDSL tcgen05 手写

**Lesson**: 负向(Triton 到顶)+ 正向基座证明。大 page256 shape 已到 Triton-on-SM103 天花板,BLOCK_N/stages/warps/splits 全都动不了 —— 差距需要手调 tcgen05 scheduling/TMA,Triton 能自动命中 tcgen05 指令但达不到手写 CUDA 的水平。规则: 自动编译器的调参空间有其硬上限,越过它必须换手写 tcgen05 path;好在同一批 tcgen05 积木(MMA(64,64,16)+TMEM accumulator+Ld16x64bOp readback)已在 SM103 落地,为 V9/V10 的 CuTeDSL 重写铺好路。

<sub>`occupancy_tuning` `tcgen05_umma` `tmem` `instruction_issue_bound` `occupancy_limited` `register_pressure`  session `6270b3b2`</sub>

