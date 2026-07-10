# kernel-experience-memory (memory-skill-wiki)

**English** · [中文](README.zh.md)

Automatically distills the scattered experience from each **GPU-kernel-optimization
session** (which methods helped/hurt, on what hardware + DSL + shapes, why, and the
measured numbers) into a **structured, reusable knowledge base**, laid out in
`atrex-kernel-agent/gpu-wiki` form so it can be merged in after human review.

> In one line: **you just optimize; this skill records the "experience" in the
> background and hands you a review-ready summary at the end.**
> Works for both **AKA (atrex-kernel-agent) autonomous runs** and **manual vibe
> coding**.

---

## The problem it solves

Kernel optimization is heavy trial-and-error: round after round of tile / warp /
pipeline / instruction changes, with the judgement of "why this worked and that
didn't" buried in the session — gone the moment you close it. Doing it by hand is
tedious and drops numbers. This skill:

- grabs new progress at **turn boundaries** and has an LLM distill it, per a
  template, into experience records that carry **real numbers**;
- files them **per operator** (`knowledge/<operator>/`), accumulating across sessions;
- generates gpu-wiki-structured docs, to be merged into the real knowledge base
  **after human review**.

---

## Quick start

```bash
# 1) One-click install: skill + three commands + global monitoring hook
bash install.sh --global
# 2) Restart the session (hooks load at startup)

# 3) In your optimization session:
/arm-run [operator]    # start monitoring (operator optional; AKA auto-detects, vibe infers at summary)
#   ... do your optimization as usual ...
/run-status            # check monitoring status anytime
/done-run              # finish: stop monitoring + produce the final summary (inline, reviewable now)
```

Install **symlinks** the skill dir + **copies** the three commands into
`~/.claude/commands/` + appends one Stop hook to `~/.claude/settings.json`
(coexisting with any existing hooks). Uninstall: `bash install.sh --uninstall`.

---

## The three commands

| Command | What it does |
|---|---|
| `/arm-run [operator]` | Turn on monitoring for the **current session** (writes an armed marker). From then on, at each turn boundary where "≥20 min since last AND there's new content", a silent background summary is written to `knowledge/<operator>/`. |
| `/run-status` | Show: armed?, operator, mode (aka/vibe), progress, how much is unprocessed, how many records accumulated, whether the background worker is running, whether the hook is installed. |
| `/done-run` | Stop monitoring + do the final summary **inline in the current session**: extract the remaining slice, consolidate, write a review-ready session summary, rebuild the local `wiki/`, and print the command to merge into gpu-wiki. |

---

## How it works

```
/arm-run ─────▶ state/armed/<sid>.json          "monitor switch" (per session)
                     │
        every turn boundary fires the Stop hook (hooks/memory_extract_hook.py)
                     │  gate: ① armed?  ② ≥ debounce (default 20min)?
                     │        ③ new content? (scripts/detect_change.py)
                     ▼  all three pass
        detached worker (claude -p, own session, reads transcript read-only)
                     │  filter+slice → attach workspace ground-truth numbers → LLM extract
                     ▼
        knowledge/<operator>/*.json              structured records (per operator, cross-session)
                     │  merge_wiki.py (serialized by a global lock)
                     ▼
        wiki/ (gpu-wiki structure, local staging)
                     │
/done-run ────▶ stop monitoring + final inline summary + session summary (awaiting review)
                     │  scripts/promote_wiki.sh (run manually after review)
                     ▼
        atrex-kernel-agent/gpu-wiki/
```

**Key mechanisms:**

- **Monitoring = the Stop hook.** Claude Code's `Stop` event fires once when the
  agent finishes a whole turn and yields back to you (not per tool call). So capture
  happens at **complete turn boundaries** — it never reads a half-written turn.
- **Trigger gate (any one fires):** a new AKA `memory/vN.json` / a perf·PASS-FAIL
  result line / a kernel `git commit` / ≥N new turns. If none, it stays quiet.
- **Two-step pipeline:** ① a script deterministically filters + segments the
  transcript (drops tool noise, keeps human prompts, assistant text/thinking, result
  signals); ② the LLM turns each "attempt worth remembering" into a record,
  following the guides in `templates/extraction/`.
- **Numbers come from the workspace, not the transcript:** latency/TFLOPS/util/
  rel_err are taken primarily from AKA's `kernel_opt_*/memory/vN.json`,
  cross-referenced with the transcript's reasoning.

---

## Key points (read these)

1. **Zero intrusion.** The hook is global, but for **un-armed sessions it's a pure
   no-op** (first thing it does is check the marker → `exit 0`; no blocking, no
   injection, no behavior change), costing only a few ms. So a global install is safe.
2. **AKA + vibe both work.** No longer requires AKA's `memory/vN.json`; vibe coding
   triggers on the result-line / commit / turn-count signals too.
3. **Numbers-first + record both signs.** Every record must carry real numbers (or,
   failing that, the exact condition it held under); negative experiences matter as
   much as positive ones — **the reason it failed is the payload.** See `SKILL.md`'s
   hard rules and controlled tag vocabulary.
4. **Human-review gate.** The automated flow only reaches the **local `wiki/`**.
   Merging into the real `atrex-kernel-agent/gpu-wiki` is a **manual** step
   (`scripts/promote_wiki.sh`) — you review, then push.
5. **Seamless re-arm.** `/arm-run` again after `/done-run` (to keep optimizing)
   resumes from the last checkpoint, processing **only new content**, still into the
   same `knowledge/<operator>/` — no special handling. This works because of three
   decoupled layers: switch (armed, per session) / progress (checkpoint, per session,
   survives done-run) / knowledge (per operator, cross-session).
6. **Concurrency-safe.** Multiple sessions monitoring different operators don't
   clash; the only shared write-back (rebuild wiki + advance checkpoint + commit) is
   serialized by a global lock, while the slow LLM extraction stays parallel.
7. **Isolation.** The background worker is a separate `claude -p` process that
   **reads** the transcript only and **writes** only under the skill's `knowledge/`/
   `state/`/`wiki/` — it never touches the AKA workspace and never recurses into the hook.

---

## Outputs

- `knowledge/<operator>/*.json` — structured experience records (git-tracked);
  schema in `SKILL.md`.
- `wiki/` — gpu-wiki-structured docs deterministically generated from the records
  (kernel-opt / pitfalls / ref-docs, by vendor→dsl→arch).
- `knowledge/<operator>/SESSION_<sid>_<date>.md` — the human-review session summary
  produced by `/done-run`.

---

## Extending the extraction/summary templates (drop-in)

Templates are **directory-scanned** — to change or add, just touch the files, no
code change / reinstall:

- `templates/extraction/*.md` — the "how to write" guides read during extraction
  (**all** of them are read).
- `templates/summary/*.md` — the `/done-run` session-summary skeletons (it picks the
  best fit).
- Rule: only `*.md` is read; **`README*` and `_`/`.`-prefixed files are ignored**
  (prefix a file with `_` to temporarily disable it).
- Editing **content** = takes effect immediately; editing a **command file**
  (`commands/*.md`) requires re-running `install.sh`.

Each template dir has its own `README.md` with details.

---

## Directory layout

```
memory-skill-wiki/
├─ README.md / README.zh.md      # this file (for humans, bilingual)
├─ SKILL.md                      # the agent-facing skill definition (schema / tags / hard rules)
├─ install.sh                    # one-click install/uninstall
├─ commands/                     # the three slash commands (installed into <.claude>/commands/)
│  ├─ arm-run.md  done-run.md  run-status.md
├─ scripts/
│  ├─ session_ctl.py            # control plane: resolve/arm/disarm/status/prep-done
│  ├─ detect_change.py          # 4-signal "new content" gate
│  ├─ extract_transcript.py     # step ①: filter + segment the transcript
│  ├─ collect_workspace.py      # pull ground-truth numbers from memory/vN.json
│  ├─ locate_session.py         # resolve a dir's session transcript
│  ├─ checkpoint.py             # per-session progress / debounce / locks (incl. global)
│  ├─ merge_wiki.py             # knowledge/ → wiki/ (gpu-wiki structure)
│  ├─ run_extraction.sh         # background worker (filter→numbers→LLM→finalize)
│  ├─ finalize_extraction.sh    # global-locked write-back (merge + advance + commit)
│  └─ promote_wiki.sh           # manual, review-gated push into gpu-wiki
├─ templates/
│  ├─ extraction/*.md           # extraction guides (auto-scanned)
│  └─ summary/*.md              # summary skeletons (auto-scanned)
├─ hooks/
│  └─ memory_extract_hook.py    # non-blocking Stop/SubagentStop hook (acts only when armed)
├─ knowledge/<operator>/*.json  # structured records (git-tracked)
├─ wiki/                        # generated docs
└─ state/                       # checkpoint.json + armed/ + locks + sessions/ + pending/
```

---

## Configuration & tuning

- **Debounce window:** default 20 min. `bash install.sh --debounce-min 30`, or the
  `MEMORY_SKILL_DEBOUNCE_MIN` env var, or per-session via the `/arm-run` marker.
- **LLM step:** the background path uses `claude -p` by default; override with
  `MEMORY_SKILL_LLM_CMD` (receives the filtered-file path as its last arg; must write
  `knowledge/*.json`) — lets you test without `claude`.
- **Install scope:** `--global` (recommended; `/arm-run` works from any dir) or
  `--claude-dir <proj>/.claude` (that project only; zero overhead elsewhere).

---

## Merging into gpu-wiki (after human review)

```bash
# see what would change
scripts/promote_wiki.sh /path/to/atrex-kernel-agent/gpu-wiki --dry-run
# then merge for real, and review the diff in the target repo before committing
scripts/promote_wiki.sh /path/to/atrex-kernel-agent/gpu-wiki
```

---

## FAQ

- **Installed but nothing happens?** Hooks load at session startup — **restart the
  session** after installing.
- **`/arm-run` says the hook isn't installed?** Run `bash install.sh --global`, then restart.
- **No records ever produced?** Probably no trigger (< debounce since last, or the
  slice has no result/commit/enough turns). Use `/run-status` to see "unprocessed
  turns" and "last summary time". Logs are in `state/extraction.log`.
- **How often does it summarize?** At most once per debounce window, and only with
  new content; rapid commits are **batched** into one (nothing lost, processed together).
- **README vs SKILL.md?** This README is for **humans**; `SKILL.md` is for the
  **agent** (record schema, controlled tags, numbers-first hard rules) — tune
  extraction quality via `SKILL.md` and `templates/extraction/`.
