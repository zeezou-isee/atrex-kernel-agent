# Real-time monitoring hook

**English** · [中文](README.zh.md)

`memory_extract_hook.py` is a **non-blocking** `Stop` (and `SubagentStop`) hook. It
fires at every turn boundary in every session, but does real work **only for
sessions armed via `/arm-run`** — for everything else it exits 0 immediately (a
pure no-op, invisible to the agent). It coexists with any other Stop hooks.

## What it does on each Stop
1. Reads the payload (`transcript_path`, `cwd`, `session_id`, `stop_hook_active`).
2. **Armed check first**: if `state/armed/<session_id>.json` is absent → exit 0
   (this single stat is what keeps it free for unrelated sessions).
3. **Debounce** (`checkpoint.py should-run`): ≥ the marker's `debounce_min`
   (default 20) minutes since the last extraction.
4. **New content** (`detect_change.py`): any of — a new AKA `memory/vN.json`, a
   perf/PASS-FAIL result line, a kernel `git commit`, or ≥ `min_turns` new turns.
5. If all pass: freeze the end uuid and launch `run_extraction.sh` fully detached
   (its own session), then exit 0. The worker distills the new slice into
   `knowledge/<operator>/`, regenerates the wiki, advances the checkpoint, and
   commits — all isolated from the running session.

## Install
Use the top-level installer (registers skill + commands + this hook):

```bash
bash ../install.sh --global          # or: --claude-dir <proj>/.claude
```

`atrex_settings_snippet.json` shows the raw `settings.json` entries if you'd rather
merge them by hand. Restart the runtime so hooks reload.

## Tuning
- `MEMORY_SKILL_DEBOUNCE_MIN` — minutes between extractions (default 20).
- `MEMORY_SKILL_LLM_CMD` — override the LLM step (receives the filtered-md path as
  its last arg; must write `knowledge/*.json`). For testing without `claude`.

## Uninstall
`bash ../install.sh --uninstall` — removes the hook entries tagged
`kernel-experience-memory-hook-v1`, the command files, and the skill link.
