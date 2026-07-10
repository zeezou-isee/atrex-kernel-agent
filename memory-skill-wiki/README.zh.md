# kernel-experience-memory (memory-skill-wiki)

[English](README.md) · **中文**

把一次次 **GPU kernel 优化会话**里散落的经验(哪些方法有效/无效、在什么硬件+DSL+形状下、
为什么、实测多少)自动**沉淀成结构化、可复用的知识库**,并整理成 `atrex-kernel-agent/gpu-wiki`
的格式,等人工审查后并入。

> 一句话:**优化的时候你只管优化;这个 skill 在后台把"经验"记下来,结束时给你一份可审查的总结。**
> 适用于 **AKA(atrex-kernel-agent)自动优化** 和 **手动 vibe coding** 两种场景。

---

## 解决什么问题

kernel 优化是高度试错的过程:一轮轮改 tile / warp / 流水 / 指令,大量"为什么这样有效、那样
不行"的判断埋在 session 里,会话一关就丢了。人工整理又累又容易漏数字。这个 skill:

- 在**回合边界**自动抓取新进展,交给 LLM 按模板提炼成一条条带**真实数字**的经验记录;
- 按**算子**归档(`knowledge/<operator>/`),跨会话持续积累;
- 生成 gpu-wiki 结构的文档,**人工审查后**再并入正式知识库。

---

## 快速开始

```bash
# 1) 一键安装:skill + 三个命令 + 全局监视 hook
bash install.sh --global
# 2) 重启 session(hook 在启动时加载)

# 3) 在你的优化 session 里:
/arm-run [算子名]      # 开始监视(算子名可省略,AKA 自动识别,vibe 由总结时推断)
#   ... 正常做你的优化 ...
/run-status            # 随时查监视状态
/done-run              # 结束:停监视 + 生成最终总结(当前会话内联,可当场审查)
```

安装是**软链** skill 目录 + **复制**三个命令到 `~/.claude/commands/` + 往 `~/.claude/settings.json`
追加一个 Stop hook(与已有 hook 并存)。卸载:`bash install.sh --uninstall`。

---

## 三个命令

| 命令 | 作用 |
|---|---|
| `/arm-run [算子名]` | 打开对**当前 session** 的监视(写一个 armed 标记)。之后每个回合边界,只要"距上次≥20 分钟且有新内容",后台就静默总结到 `knowledge/<算子>/`。 |
| `/run-status` | 查看:是否被监视、算子、模式(aka/vibe)、处理进度、还有多少未处理、已积累几条记录、后台是否在跑、hook 是否装好。 |
| `/done-run` | 关闭监视 + 在**当前会话内联**做最终总结:抽取剩余切片、合并去重、写一份待审查的会话总结、重建本地 `wiki/`,并打印手动并入 gpu-wiki 的命令。 |

---

## 工作原理

```
/arm-run ─────▶ state/armed/<sid>.json          「监视开关」(每会话)
                     │
        每个回合结束触发 Stop hook(hooks/memory_extract_hook.py)
                     │  闸门:① 被 arm 了吗  ② 距上次 ≥ debounce(默认20min)吗
                     │        ③ 有新内容吗(scripts/detect_change.py)
                     ▼  三者都满足
        后台独立 worker(claude -p,独立会话,只读转录)
                     │  过滤+切片 → 附上 workspace 权威数字 → LLM 按模板抽取
                     ▼
        knowledge/<operator>/*.json              结构化记录(每算子,跨会话)
                     │  merge_wiki.py(全局锁串行)
                     ▼
        wiki/(gpu-wiki 结构,本地暂存)
                     │
/done-run ────▶ 关监视 + 最终内联总结 + 会话总结(待人工审查)
                     │  scripts/promote_wiki.sh(审查后手动执行)
                     ▼
        atrex-kernel-agent/gpu-wiki/
```

**关键机制:**

- **监视 = Stop hook**。Claude Code 的 `Stop` 事件在"agent 干完一整轮、回到等你输入"时触发一次
  (不是每个工具调用)。所以抓取天然发生在**完整回合边界**,不会读到写了一半的对话。
- **触发闸门(4 选 1 即触发)**:新的 AKA `memory/vN.json` / 性能·PASS-FAIL 结果行 /
  kernel 的 `git commit` / ≥N 个新回合。都不满足就不打扰。
- **两步流水线**:① 脚本确定性地过滤+切分转录(丢掉工具噪音,保留人类提问、assistant 文本/思考、
  结果信号);② LLM 按 `templates/extraction/` 里的指南把每个"值得记的尝试"写成一条记录。
- **数字来自 workspace,不是转录**:latency/TFLOPS/利用率/rel_err 优先取自 AKA 的
  `kernel_opt_*/memory/vN.json`,与转录里的说法交叉印证。

---

## 重点(务必知道这几条)

1. **零侵入**:hook 全局安装,但对**没 arm 的 session 是纯空操作**(第一步就查标记,`exit 0`,
   不阻塞、不注入、不改任何行为),只多几十毫秒。所以挂全局很安全。
2. **AKA + vibe 通用**:不再依赖 AKA 的 `memory/vN.json` 才能工作;vibe coding 靠"结果行/commit/
   回合数"信号也能触发。
3. **数字优先 + 正负都记**:每条记录必须带真实数字(没有就写死成立条件);失败经验和成功经验
   同等重要——**失败的原因就是价值**。详见 `SKILL.md` 的 Hard rules 与受控标签词表。
4. **人工审查门**:自动流程只到**本地 `wiki/`**。并入正式的 `atrex-kernel-agent/gpu-wiki` 是
   **手动**一步(`scripts/promote_wiki.sh`),你审完再推。
5. **re-arm 无缝续跑**:`/done-run` 之后再 `/arm-run`(继续优化),会从上次 checkpoint 处**只处理
   新内容**,仍进同一个 `knowledge/<算子>/`——不需要任何特殊操作。原理是三层解耦:
   开关(armed,会话级)/ 进度(checkpoint,会话级,done 后保留)/ 知识(按算子,跨会话)。
6. **并发安全**:多个 session 同时监视不同算子互不冲突;唯一的共享写回段(重建 wiki+推进
   checkpoint+提交)用全局锁串行,慢的 LLM 抽取仍各自并行。
7. **隔离**:后台 worker 是独立 `claude -p` 进程,**只读**转录、只写本 skill 的
   `knowledge/`/`state/`/`wiki/`,**绝不碰 AKA 工作区**,也不会递归触发自己。

---

## 产出物

- `knowledge/<operator>/*.json` — 结构化经验记录(git 跟踪),schema 见 `SKILL.md`。
- `wiki/` — 由记录确定性生成的 gpu-wiki 结构文档(kernel-opt / pitfalls / ref-docs,
  按 vendor→dsl→arch)。
- `knowledge/<operator>/SESSION_<sid>_<date>.md` — `/done-run` 产出的、给人审查的会话总结。

---

## 扩展提取/总结模板(drop-in)

模板是**扫描目录**的,想改想加直接动文件即可,无需改代码/重装:

- `templates/extraction/*.md` — 抽取记录时的"怎么写"指南,**全部**会被读取。
- `templates/summary/*.md` — `/done-run` 的会话总结骨架,`/done-run` 从中挑最合适的。
- 规则:只读 `*.md`,**忽略 `README*` 及以 `_`/`.` 开头的文件**(想临时停用某个模板,加 `_` 前缀)。
- 改**内容** = 即时生效;改**命令文件本身**(`commands/*.md`)才需要重跑 `install.sh`。

每个模板目录下的 `README.md` 有更细的说明。

---

## 目录结构

```
memory-skill-wiki/
├─ README.md / README.zh.md      # 本文件(给人读,中英双版)
├─ SKILL.md                      # 给 agent 读的 skill 定义(schema / 标签 / 硬规则)
├─ install.sh                    # 一键安装/卸载
├─ commands/                     # 三个斜杠命令(装到 <.claude>/commands/)
│  ├─ arm-run.md  done-run.md  run-status.md
├─ scripts/
│  ├─ session_ctl.py            # 控制面:resolve/arm/disarm/status/prep-done
│  ├─ detect_change.py          # 4 信号"新内容"闸门
│  ├─ extract_transcript.py     # 第①步:过滤+切分转录
│  ├─ collect_workspace.py      # 抓 memory/vN.json 的权威数字
│  ├─ locate_session.py         # 解析目录对应的 session 转录
│  ├─ checkpoint.py             # 每会话进度/去抖/锁(含全局锁)
│  ├─ merge_wiki.py             # knowledge/ → wiki/(gpu-wiki 结构)
│  ├─ run_extraction.sh         # 后台 worker(过滤→数字→LLM→finalize)
│  ├─ finalize_extraction.sh    # 全局锁写回段(合并+推进+提交)
│  └─ promote_wiki.sh           # 审查后手动推 gpu-wiki
├─ templates/
│  ├─ extraction/*.md           # 提取指南(自动扫描)
│  └─ summary/*.md              # 总结骨架(自动扫描)
├─ hooks/
│  └─ memory_extract_hook.py    # 非阻塞 Stop/SubagentStop hook(armed 才动作)
├─ knowledge/<operator>/*.json  # 结构化记录(git 跟踪)
├─ wiki/                        # 生成的文档
└─ state/                       # checkpoint.json + armed/ + 锁 + sessions/ + pending/
```

---

## 配置与调参

- **去抖窗口**:默认 20 分钟。`bash install.sh --debounce-min 30`,或环境变量
  `MEMORY_SKILL_DEBOUNCE_MIN`,或单会话在 `/arm-run` 时按算子存进标记。
- **LLM 步骤**:后台默认用 `claude -p`;可用 `MEMORY_SKILL_LLM_CMD` 覆盖(收到 filtered 文件路径作
  最后一个参数,须写出 `knowledge/*.json`)——测试时不依赖 `claude` 也能跑。
- **安装范围**:`--global`(推荐,任何目录都能 `/arm-run`)或 `--claude-dir <proj>/.claude`(仅该项目,
  其它 session 零开销)。

---

## 并入 gpu-wiki(人工审查后)

```bash
# 先看会改动什么
scripts/promote_wiki.sh /path/to/atrex-kernel-agent/gpu-wiki --dry-run
# 确认后正式并入,再到目标仓库 review diff 后提交
scripts/promote_wiki.sh /path/to/atrex-kernel-agent/gpu-wiki
```

---

## 常见问题

- **装完为什么不生效?** hook 在 session 启动时加载,装完要**重启 session**。
- **`/arm-run` 提示 hook 未安装?** 跑 `bash install.sh --global` 再重启。
- **一直没产出记录?** 可能是没触发(距上次 <20min,或切片里没有结果/commit/足够回合);
  用 `/run-status` 看"未处理回合数"和"上次总结时间"。日志在 `state/extraction.log`。
- **多久总结一次?** 至少间隔去抖窗口,且要有新内容;频繁 commit 会被**合并**成一次(不丢,攒着一起处理)。
- **README 和 SKILL.md 什么关系?** 本 README 给**人**读;`SKILL.md` 给 **agent** 读(记录 schema、
  受控标签、numbers-first 硬规则)——要改抽取质量看 `SKILL.md` 和 `templates/extraction/`。
