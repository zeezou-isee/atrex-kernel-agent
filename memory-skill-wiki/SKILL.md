---
name: kernel-experience-memory-wiki
description: >
  gpu-wiki-structured variant of kernel-experience-memory. Extracts reusable
  GPU-kernel-optimization experience (positive AND negative) from Claude Code
  session transcripts into a structured, tagged knowledge base, then merges it
  into a wiki whose layout MATCHES atrex-kernel-agent/gpu-wiki (kernel-opt /
  pitfalls / ref-docs, vendor→dsl→arch) so it can feed that knowledge base. Use
  when the user asks to extract/summarize kernel-optimization experience and wants
  the final wiki in gpu-wiki form, or in the atrex real-time background flow.
---

# Kernel-Experience Memory (gpu-wiki-structured variant)

> This is the **`memory-skill-wiki`** variant of `memory-skill`. Everything is
> identical EXCEPT the final wiki form: `merge_wiki.py` emits gpu-wiki-structured
> docs (see "Merge into wiki" below) instead of the base skill's flat wiki.

This skill turns the messy transcript of a kernel-optimization session into a
distilled, queryable memory of **what optimization was tried, on what hardware /
DSL, why, and what actually happened — with numbers**. It captures both wins and
dead-ends, because knowing what *failed* (and why) is as valuable as what worked.

It pairs with `atrex-kernel-agent` (AKA): AKA *does* the optimization and writes
per-iteration `memory/v<N>.json`; this skill *distills* the reasoning around those
iterations into cross-iteration, tagged experience that survives the session.

## Three entry points

1. **Offline (user-invoked).** The user points at one or more complete session
   `.jsonl` files and asks to extract experience. Process the whole file(s).
2. **Real-time (background, armed session).** After `/arm-run`, a non-blocking
   `Stop` hook (`hooks/memory_extract_hook.py`) launches a detached headless
   process that runs THIS skill on only the *new* slice since the last checkpoint.
   It never blocks or injects into the main agent — updates are silent and low
   frequency (debounced ~20 min; fires on a new `memory/v<N>.json`, a perf/PASS-FAIL
   result line, a kernel commit, or ≥N new turns). Works for AKA **and** vibe coding.
3. **Final (user-invoked, `/done-run`).** Stops monitoring and does an inline
   consolidation pass over the remaining slice, writes a review-ready session
   summary, and stages the local `wiki/` for manual promotion into gpu-wiki.

All paths use the same two-step pipeline and write to the same per-operator
`knowledge/<operator>/`. See "Control-plane commands" below.

## Control-plane commands (arm-run / done-run / run-status)

Three slash commands (installed by `install.sh` into `<.claude>/commands/`) drive
monitoring; each resolves the current session via `scripts/session_ctl.py`.

- **`/arm-run [operator]`** — monitoring ON for this session. Writes
  `state/armed/<session_id>.json` (cwd, operator, mode, debounce, min_turns).
  operator = the arg if given, else auto-detected from a `kernel_opt_<name>` folder
  (AKA), else `null` (summarizer infers it, filing under `knowledge/_inbox/` until
  `/done-run` reconciles). The Stop hook is a pure no-op for un-armed sessions.
- **`/run-status`** — report armed?/operator/mode/checkpoint/unprocessed-slice/
  record-count/worker/hook for this session.
- **`/done-run`** — monitoring OFF + final inline summary: `session_ctl.py
  prep-done` disarms + freezes + builds the filtered slice; the agent then
  writes/consolidates `knowledge/<operator>/*.json` (per **all** guides in
  `templates/extraction/`), writes a review summary from a skeleton in
  `templates/summary/`, and runs `scripts/finalize_extraction.sh`
  (global-locked merge + checkpoint advance + commit). Promotion into
  `atrex-kernel-agent/gpu-wiki` stays **manual** (`scripts/promote_wiki.sh`) — the
  human-review gate.

**State model (why re-arm after done-run just works).** Three decoupled layers:
`state/armed/<sid>.json` = "monitoring on?" (per session; arm creates, done-run
deletes); `state/checkpoint.json[sid]` = "how far extracted" (per session; survives
done-run); `knowledge/<operator>/` = accumulated knowledge (per operator,
cross-session). Re-running `/arm-run` after `/done-run` resumes extraction `--since`
the last checkpoint into the same operator folder — no special-casing.

**Gate & isolation.** The hook checks the armed marker FIRST (cheap no-op
otherwise), then debounce (`checkpoint.py should-run`), then content
(`scripts/detect_change.py`). The detached worker runs in a separate *unarmed*
`claude -p` session, reads the transcript read-only, writes only under
`knowledge/`/`state/`/`wiki/`, and never touches the AKA workspace (so it also
never recurses into the hook).

## Pipeline (always two steps)

### Step 1 — deterministic filtering & segmentation (script)

Run the extractor to get clean, segmented text. Never hand-parse the raw jsonl.

```bash
# Don't know the path? Resolve the current dir's transcript:
SESSION=$(python3 scripts/locate_session.py --cwd /path/to/atrex/run)

# offline, whole file:
python3 scripts/extract_transcript.py "$SESSION" --out /tmp/filtered.md

# incremental (real-time): only [checkpoint, frozen-end], see checkpoint.py below
FREEZE=$(python3 scripts/checkpoint.py freeze "$SESSION")
SINCE=$(python3 scripts/checkpoint.py since <SESSION_ID>)
python3 scripts/extract_transcript.py "$SESSION" \
    ${SINCE:+--since "$SINCE"} --freeze-at "$FREEZE" --out /tmp/filtered.md

# ALWAYS also attach the atrex workspace ground-truth numbers (see below):
python3 scripts/collect_workspace.py --cwd <atrex-cwd> --since-version <N> >> /tmp/filtered.md
```

What the script guarantees (so you don't have to re-check):
- Drops all `tool_use` inputs; keeps human prompts, assistant `text`, assistant
  `thinking`, a compact per-turn tool tally, AND **result-signal lines** salvaged
  from `tool_result` (perf numbers / PASS-FAIL / errors only — noise dumps are
  dropped). ~70% volume cut while preserving 产出的结果.
- Splits into **turns** at each real human message (`promptSource` typed/queued/
  suggestion_accepted). One turn may contain several kernel versions.
- Inlines **work** sub-agents (baseline/optimizer/Explore/...) under the turn that
  spawned them — including their **task prompt** and result signals — and
  **excludes this skill's own** extraction sub-agents by agentType/description.
- `--since`/`--freeze-at` slice the append-only chain; the frozen end is captured
  first, so content appended mid-extraction is simply left for next time.

**Authoritative numbers come from the atrex workspace, not the transcript.** Large
tool outputs are offloaded to files and benchmark/ncu numbers live in the tool
layer, so the transcript prose is an unreliable source of figures. atrex writes the
ground truth into `kernel_opt_*/memory/v<N>.json` (latency_us, tflops, util %,
rel_err, action_category, expected_impact, pitfalls) and `README.md` (sourced
Hardware Spec + Stop Conditions). `collect_workspace.py` digests exactly those and
appends them to the filtered text. **Prefer these numbers as authoritative** in
Step 2, cross-referenced with the transcript's reasoning.

### Step 2 — LLM distillation into structured records (you)

Read `/tmp/filtered.md`. It is already clean. Follow **every guide in
`templates/extraction/`** (能写 / 必写 / 不写 / 怎么写 + good-vs-bad examples — they
constrain form, not content) together with the schema below. Now, **within each
turn / logical unit**, identify the distinct *attempts* (a "unit" = one human turn, OR one
produced kernel version, OR one staged result — the hybrid definition; a turn may
hold several attempts, or an attempt may span a couple of "继续" turns — use
judgment). For **each attempt worth remembering**, emit one experience record.

An attempt is "worth remembering" if it teaches something transferable: a method
that helped or hurt a specific bottleneck, a pitfall + root cause + fix, a
hardware/DSL constraint that shaped the design. Skip pure logistics (env setup,
file moves, path confusion) — unless they encode a real gotcha.

Write each record as `knowledge/<slug>.json` following the schema below. Then run
`scripts/merge_wiki.py` to (re)generate the wiki.

## Hard rules

1. **Numbers first.** `attempt.measured` MUST carry concrete figures. Draw them
   primarily from the workspace digest (`memory/v<N>.json`): latency (us), TFLOPS,
   GB/s, utilization %, rel_err, occupancy, delta vs the previous version. If a
   claim genuinely has no number anywhere, you MUST instead state the exact
   workload/condition it held under (shape, dtype, block size, seqlen…). Never
   write a vague "faster/better" with neither. Tie each record to its `version`
   and `git_commit` when the digest provides them.
2. **Both signs.** Record negative experiences (`outcome: negative`) with the same
   rigor as positive ones — the *reason* it failed is the payload.
3. **Attribute.** `attempt.reason` must say *why* it worked/failed, grounded in the
   transcript (evidence → inference), not a generic platitude.
4. **No fabrication.** Only record what the transcript supports. If a number is
   ambiguous, mark it (e.g. `~`, or note "unverified in transcript"). Do not invent
   hardware specs — those belong to atrex's gpu-wiki.
5. **Stay isolated.** In real-time mode you run in a detached process/sub-agent.
   Only write under `knowledge/`, `wiki/`, `state/`. Never touch the AKA workspace,
   never print into the main agent's context, always let the hook exit 0.
6. **Advance the checkpoint** only after records are written & committed:
   `python3 scripts/checkpoint.py advance <SID> --last-uuid <FREEZE> --mem-version <V>`.

## Experience record schema (`knowledge/<slug>.json`)

```jsonc
{
  "id": "paged-attn_b300_in-kernel-pv-fusion_v10",  // stable, descriptive slug
  "source": {
    "session_id": "6270b3b2-...",
    "turn_range": "[23,25]",           // turns in /tmp/filtered.md this came from
    "git_commit": "9a14c6d",           // AKA workspace commit if known, else null
    "extracted_at": "2026-07-03T11:00:00Z"
  },
  "context": {
    "operator": "paged_attention_decode",
    "hardware": "B300 / sm103",        // free text; also mirrored into tags
    "dsl": "CuTeDSL (cutlass 4.5.2)",
    "dtype": "bf16 in/out, fp32 accum",
    "shapes": "hd256, GQA nqh=16, q_per_seq=4, block_n=64",
    "workload": "decode stage"
  },
  "attempt": {
    "method": "in-kernel P@V fusion, 2-CTA tile, TMEM accumulation",
    "expected": "cut HBM round-trips, raise Tensor-Core utilization",
    "measured": "latency 120us -> 95us (-21%); TC util 62% -> 74%; rel_err 3e-3 PASS",
    "reason": "PV kept in TMEM avoided the SMEM staging round-trip that ncu showed as the top warp-stall; 2-CTA needed to fit hd256 columns",
    "outcome": "positive"              // positive | negative | neutral
  },
  "tags": {
    "hardware": {"platform": ["B300"], "arch": ["Blackwell"], "target": ["sm103"]},
    "bottleneck": ["memory_bound", "pipeline_stall"],
    "category": {"group": ["instruction", "algorithm"],
                 "technique": ["in_kernel_fusion", "tmem", "cta_cooperation"]}
  },
  "body": "<five-part narrative, see template>"
}
```

## Summary template — five parts (`body`, declarative prose)

Write `body` as five short labeled paragraphs. Declarative, specific, transferable.

1. **工况 (Setting):** operator + hardware + DSL + dtype/shape + which stage.
2. **方法 (Method):** exactly what was changed (tile / warp / overlap / instruction…).
3. **预期 (Expectation):** which bottleneck it targeted and the mechanism, why.
4. **实测 (Result):** the measured effect WITH NUMBERS (latency/TFLOPS/BW-util/rel_err);
   if no number exists, the precise condition under which the behavior held.
5. **归因与结论 (Attribution & takeaway):** why it succeeded/failed, positive or
   negative, and the actionable rule for next time.

## Tag vocabulary (controlled, v3)

Three orthogonal facets, all multi-valued. Use these terms; if none fits, coin
`x-<name>` and it will be reviewed/promoted at consolidation. Lowercase +
underscores, except platform/arch keep conventional casing (`B300`, `Blackwell`).

**hardware** — structured; always give at least `platform` + `arch`:
```jsonc
"hardware": { "platform": [...], "arch": [...], "target": [...] }
```
- platform: `A100` `H20` `H100` `H200` `H800` `B200` `B300` `GB200` `GB300` `MI250X` `MI300X` `MI308X` `MI325X` `MI355X`
- arch: `Ampere` `Ada` `Hopper` `Blackwell` `CDNA2` `CDNA3` `CDNA4`
- target (compute cap): `sm80` `sm86` `sm89` `sm90` `sm90a` `sm100` `sm100a` `sm103` `sm120` `gfx90a` `gfx942` `gfx950`
- (memory generation like HBM3e goes in `context.hardware`, not tags)

**bottleneck** — flat list; tag the problem the attempt targeted AND what profile diagnosed:
- Roofline: `memory_bound` `compute_bound` `latency_bound`
- Occupancy/resource: `occupancy_limited` `register_pressure` `register_spill` `smem_lds_capacity`
- Memory subsystem: `bank_conflict` `l2_cache_bound` `uncoalesced_access`
- Pipeline/issue: `pipeline_stall` `instruction_issue_bound` `barrier_sync`
- Grid/launch: `grid_underutilization` `tail_effect` `launch_overhead`
- Numerics: `numerical_instability`

**category** — structured (`group` → `technique`); pick the group(s) and the specific technique(s):
```jsonc
"category": { "group": [...], "technique": [...] }
```
| group | technique |
|---|---|
| `tiling` | `tiling` `split_k` `split_kv` `persistent_kernel` |
| `warp` | `warp_specialization` `register_blocking` `warp_reduction` |
| `memory_movement` | `vectorized_load` `async_copy` `multi_buffering` `overlap_pipelining` `swizzle` `paged_gather` |
| `instruction` | `mma_scheduling` `tcgen05_umma` `tmem` `cta_cooperation` `fast_math` |
| `algorithm` | `in_kernel_fusion` `online_softmax` `quantization` `numerics_fix` |
| `launch` | `launch_optimization` `occupancy_tuning` |

atrex `action_category` maps in: software_pipeline→`overlap_pipelining`,
launch_overhead_reduction→`launch_optimization`, double_buffering→`multi_buffering`,
tcgen05_*→`tcgen05_umma`/`tmem`; `vectorized_load`/`swizzle` are shared.

## Merge into wiki — gpu-wiki-structured (this variant)

This is the **`memory-skill-wiki`** variant: the final wiki is generated in the exact
layout and document shapes of `atrex-kernel-agent/gpu-wiki`, so it can be dropped
straight into that knowledge base. The per-record JSON (the intermediate memory) is
unchanged — only the final `wiki/` form differs from the base skill.

**1. Deterministic render (every extraction).**

```bash
python3 scripts/merge_wiki.py            # regenerate wiki/ from knowledge/
```

It routes each record by `vendor → dsl → arch` (from the tags) and, per `topic`
(= operator), emits up to three docs mirroring gpu-wiki:

```
wiki/README.md                                                     top routing table
wiki/docs/kernel-opt/<vendor>/<dsl>/<arch>/<topic>.md              positives → Trigger + technique set
wiki/docs/pitfalls/<vendor>/<dsl>/<arch>/<topic>-pitfalls.md       negatives → Trap / Result / Why / Lesson (5-step)
wiki/docs/ref-docs/<vendor>/<dsl>/<arch>/<topic>-optimization.md   all → version-ladder journey
wiki/docs/{kernel-opt,pitfalls}/<vendor>/<dsl>/<arch>/README.md    index tables (File | Kernel | Hardware | count)
```

The five-part record `body`/fields map onto gpu-wiki sections:
`方法`→**Trap**/**Technique**, `预期`→**Expected**, `实测`→**Result**/**Effect**,
`归因与结论`→**Why**/**Lesson**. Version numbers are parsed from the record `id`
(`…_vN_…`) to build the ref-docs version ladder. Deterministic (dates come from
`source.extracted_at`, not the clock); same records → same bytes.

**2. LLM consolidation (session end / on request).** When the AKA run reaches its
Stop Conditions, or the user asks to "consolidate", do a distillation pass:
- Read all `knowledge/*.json` for the operator+platform.
- Merge **near-duplicate** attempts (same method across adjacent versions) into one
  record that shows the *trajectory* with numbers (e.g. v0 2399us → v2 100us → v4
  47us), keeping the strongest attribution and every distinct pitfall.
- Delete the merged-away files, rewrite the survivor(s), rerun `merge_wiki.py`.
- The goal is a wiki that reads as a coherent, de-duplicated knowledge base — the
  "把中间过程的结果合并总结" pass — not just a pile of per-trigger records.
This step is LLM-driven (judgment); merge_wiki.py stays the deterministic renderer.

## Files

- `install.sh` — one-click: skill + commands + hook (default) / `--without-hook` / `--global` / `--uninstall`.
- `commands/{arm-run,done-run,run-status}.md` — slash commands (copied into `<.claude>/commands/`).
- `scripts/session_ctl.py` — control plane: resolve / arm / disarm / status / prep-done.
- `scripts/detect_change.py` — 4-signal "new content" gate (version / result / commit / turns).
- `scripts/extract_transcript.py` — Step 1 filter/segment/inline (+result signals).
- `scripts/collect_workspace.py` — digest authoritative numbers from `memory/v*.json` + README.
- `scripts/locate_session.py` — resolve a dir's session transcript path.
- `scripts/checkpoint.py` — per-session checkpoint, freeze, debounce gate, memversion, lock (incl. `__global__`).
- `scripts/merge_wiki.py` — knowledge/ → wiki/ in **gpu-wiki structure** (recursive over `knowledge/<op>/`).
- `scripts/run_extraction.sh` — detached worker: lock→filter→digest→LLM→history→finalize.
- `scripts/finalize_extraction.sh` — global-locked tail: merge→advance→commit (shared by worker + done-run).
- `scripts/promote_wiki.sh` — manual, review-gated copy of `wiki/docs/*` into a target gpu-wiki.
- `templates/extraction/*.md` — Step-2 how-to guides (能写/必写/不写/怎么写); **all auto-scanned** — drop a `.md` in to extend (README + `_`/`.`-prefixed ignored).
- `templates/summary/*.md` — review-ready session-summary skeleton(s) for `/done-run`; **all auto-scanned** (same rule).
- `hooks/memory_extract_hook.py` — non-blocking Stop/SubagentStop hook (armed-gated).
- `knowledge/<operator>/` — structured records (git-tracked). `wiki/` — generated docs.
- `state/` — `checkpoint.json` + `armed/<sid>.json` + locks + `sessions/<id>.md` history + `pending/`.
