<!--
session_summary.md — TEMPLATE for the human-review summary that /done-run writes.

The agent fills this in and saves it as:
  knowledge/<operator>/SESSION_<session_id_short>_<YYYY-MM-DD>.md

This file is the human-review gate: a person reads it, ticks the checklist, then
runs scripts/promote_wiki.sh to stage the docs into atrex-kernel-agent/gpu-wiki.
Keep it tight and skimmable — numbers over prose. Delete these comments.
-->

# Session summary — `<operator>` on `<hardware>`

- **Operator / stage**: `<operator>` · `<prefill|decode|...>`
- **Hardware / DSL / dtype**: `<B300 / sm103>` · `<CuTeDSL 4.5.2>` · `<bf16 in/out, fp32 accum>`
- **Shapes**: `<hd256, GQA nqh=16, block_n=64, ...>`
- **Session / mode / date**: `<sid[:8]>` · `<aka|vibe>` · `<YYYY-MM-DD>`
- **Rounds captured this run**: `<N attempts>` → `knowledge/<operator>/`

## 一句话结论 (TL;DR)
<one sentence: where it started, where it ended, the single most important lever.>

## 版本阶梯 (trajectory, numbers first)
| Ver | Outcome | Technique | Measured (latency / util / rel_err) |
|-----|---------|-----------|-------------------------------------|
| v0  | ➖ base | dense baseline | 2399us · — · PASS |
| v2  | ✅      | <...>          | 100us (-96%) · <...> · PASS |
| v4  | ✅      | <...>          | 47us · <...> · PASS |
| vN  | ❌      | <...>          | <why it regressed / failed> |

## 关键正向经验 (wins — becomes kernel-opt docs)
- **<technique>** — <what it did, with numbers, and why it worked>.
- ...

## 关键坑 (pitfalls — becomes pitfalls docs)
- **<trap>** → <result> → <root cause> → **下次规则**: <actionable lesson>.
- ...

## 尚未解决 / 下一步 (open questions)
- <the ceiling hit, the untested idea, the unexplained number>.

## 产出的 records (this run)
- `knowledge/<operator>/<slug>.json` — <one-line each>
- ...

## 审查清单 (reviewer — tick before promoting)
- [ ] 数字都可信(与 `memory/vN.json` 或 do_bench 对得上)?
- [ ] 正/负经验都收了,归因是机制而非套话?
- [ ] tags 正确(platform/arch/bottleneck/category 都在词表内)?
- [ ] 无编造的硬件规格 / commit / 数字?
- [ ] 本地 `wiki/` 已用 `merge_wiki.py` 重建并检查?
- [ ] **准备好提交到 gpu-wiki** → 运行:
      `scripts/promote_wiki.sh <path-to>/atrex-kernel-agent/gpu-wiki --dry-run` 先看 diff,再去掉 `--dry-run`。
