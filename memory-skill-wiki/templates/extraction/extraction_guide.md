# 抽取模板 · Extraction Guide (Step 2)

> 读 `/tmp/…/*_final.md`(或 `state/pending/*.md`)这份已过滤的切片后,按本指南把每个
> "值得记的尝试"写成一条 `knowledge/<operator>/<slug>.json`。

**核心折中(务必理解):本模板只约束"怎么写"(形式与质量门槛),不约束"写什么"(内容与结论)。**
值得记什么、结论是什么,由你判断;模板只保证你记得**规范、带数字、可复用、正负都留**。
每个字段下给出四类硬约束 —— **能写 / 必写 / 不写 / 怎么写**,并配 ✅好例 / ❌坏例。
凡例子仅示范"形式",不要照抄其结论。

---

## 0. 什么值得记(先过这一关)

- **必记**:一个方法对某个瓶颈**有效或有害**且能迁移;一个坑 + 根因 + 修复;一个塑造了设计的
  硬件/DSL 约束(如 TMEM 容量、2-CTA 限制);一次"看似该有效却没效"的反直觉结果。
- **可记**:一个被数字证否的假设(负经验,和正经验同等重要)。
- **不记**:纯环境/路径/文件搬运/编译琐事 —— **除非**它本身是一个会再次坑到人的 gotcha。
- 判据一句话:**"下次换个人/换个 session,这条能不能让他少走弯路?"** 不能就别写。

---

## 1. 记录粒度:一条 record = 一次 attempt

- 一个"attempt" = 一次人类回合 **或** 一个产出的 kernel 版本 **或** 一个阶段性结果(混合定义,用判断)。
- 一个回合可能含多次 attempt;一次 attempt 也可能跨几个"继续"回合 —— 按"一个可迁移的教训"切分,别按行数切。
- **不要**把同一方法在相邻版本上的重复各写一条;那属于 done-run 的合并阶段(见 §4)。

---

## 2. 逐字段指南

### `id`(稳定、描述性的 slug)
- **必写**:能一眼看出"算子_平台_方法[_版本]"。含版本时用 `_vN`,merge_wiki 会据此排版本阶梯。
- **不写**:日期、session id、随机串、纯序号(`rec1`)。
- **怎么写**:小写 + 下划线/连字符。
- ✅ `paged-attn_b300_in-kernel-pv-fusion_v10`   ❌ `note_2026_07_09_final`

### `source`
- **必写**:`session_id`;有则写 `git_commit`(AKA workspace 的 commit)、`turn_range`、`extracted_at`(用切片里的日期,不要用"今天"随手编)。
- **不写**:编造的 commit / 行号。
- **怎么写**:拿不准的留 `null`,不要瞎填。

### `context`(工况)
- **必写**:`operator`、`hardware`(如 `B300 / sm103`)、`dsl`、`dtype`、`shapes`、`workload`(prefill/decode…)。
- **不写**:泛泛的 "GPU"、"attention";要具体到芯片/计算能力/形状。
- **怎么写**:形状写全(hd、GQA、block_n、seqlen…),这是"没有数字时"退而求其次的锚点。
- ✅ `"shapes": "hd256, GQA nqh=16, q_per_seq=4, block_n=64"`   ❌ `"shapes": "large"`

### `attempt.method`(做了什么改动)
- **必写**:精确到 tile / warp / 流水 / 指令 / 算法这一层的**具体改动**。
- **不写**:"优化了 kernel""调了参数"这种没信息量的话。
- ✅ `"in-kernel P@V fusion, 2-CTA tile, TMEM accumulation"`
  ❌ `"made it faster by tuning"`

### `attempt.expected`(想解决哪个瓶颈、机理)
- **必写**:瞄准的瓶颈 + 为什么这个改动能动它(机理假设)。
- **怎么写**:一句"因为 X 所以预期 Y"。
- ✅ `"cut HBM round-trips; keep P@V in TMEM to remove the SMEM staging stall ncu flagged"`

### `attempt.measured`(实测,**数字优先**)
- **必写(硬门槛)**:具体数字 —— latency(us)、TFLOPS、GB/s、利用率%、rel_err、occupancy、Δvs 上一版。
  **数字优先取自 workspace digest(`memory/vN.json`)**,与转录里的说法交叉印证。
- **没有任何数字时**:**必须**写清它成立的确切条件(shape / dtype / block / seqlen),并标注
  "unverified in transcript";**绝不**写没数字也没条件的"更快/更好"。
- **不写**:含糊的 "significantly faster";主观形容词。
- ✅ `"latency 120us -> 95us (-21%); TC util 62% -> 74%; rel_err 3e-3 PASS"`
  ❌ `"much faster, looks correct"`

### `attempt.reason`(归因,为什么成/败)
- **必写**:证据 → 推断的因果,落到具体机制(ncu 显示的 stall、寄存器溢出、bank conflict…)。
- **不写**:套话("因为优化了所以变快了")。
- ✅ `"PV in TMEM removed the SMEM round-trip that ncu showed as the top warp-stall; 2-CTA needed to fit hd256 cols"`

### `attempt.outcome`
- **必写**:`positive` | `negative` | `neutral`。负经验照记,**失败的原因就是价值**。

### `tags`(受控词表,见 SKILL.md)
- **必写**:`hardware.platform` + `hardware.arch`(至少这两项);`bottleneck`(≥1);`category.group`+`category.technique`。
- **不写**:自造词——除非确实没有合适的,才用 `x-<name>` 留待合并时评审。
- **怎么写**:只用 SKILL.md 的词表;大小写按约定(`B300`、`Blackwell`,其余小写下划线)。

### `body`(五段式叙述)——见 §3

---

## 3. 五段式 `body`:能写/必写/不写/怎么写

用五个带**中文标签**的短段(merge_wiki 靠这些标签切分,标签**必须保留**):
`工况:` `方法:` `预期:` `实测:` `归因与结论:`

| 段 | 必写 | 不写 | 怎么写 |
|---|---|---|---|
| **工况** | 算子+硬件+DSL+dtype/shape+阶段 | 泛化描述 | 一句锚定"在什么条件下" |
| **方法** | 到 tile/warp/指令层的具体改动 | "调优了" | 陈述句,具体 |
| **预期** | targeted 瓶颈 + 机理 | 事后诸葛 | "针对 X,预期 Y,因为 Z" |
| **实测** | **数字**(latency/TFLOPS/util/rel_err);无数字则写死条件 | 形容词 | 带单位、带对比、带 PASS/FAIL |
| **归因与结论** | 成/败的机制 + **下次可执行的规则** | 空泛感想 | 落到"下次遇到 X 就 Y" |

✅ 好例(实测段):
`实测:full-shape decode,bf16,latency 2399us → 100us(v0→v2,-96%),再到 47us(v4);rel_err 3e-3 PASS。`
❌ 坏例(实测段):
`实测:性能提升很明显,基本正确。`

---

## 4. done-run 的合并(consolidation)——仅在最终总结时做

把**同一方法跨相邻版本**的近重复合并成一条,展示**轨迹**(如 v0 2399us → v2 100us → v4 47us),
保留最强的归因和**每一个不同的坑**;删掉被合并的文件,重写幸存者;`_inbox/` 里的记录归位到
`knowledge/<operator>/`。目标是一份去重、连贯、可直接读的知识库,而不是一堆按触发点堆的碎片。

---

## 5. 提交前自查(坏味道清单)

- [ ] `measured` 里有真数字,或(退一步)写死了成立条件 + `unverified` 标注?
- [ ] `reason` 是"证据→机制",不是套话?
- [ ] 负经验也照规格记了(`outcome: negative`)?
- [ ] tags 至少有 platform+arch+bottleneck+category,且都来自受控词表?
- [ ] `id` 稳定、含版本(如适用)?文件在 `knowledge/<operator>/` 下?
- [ ] 没有编造硬件规格 / commit / 数字?拿不准的都标了 `~` 或 `unverified`?
