---
description: End monitoring for THIS session and produce the final, review-ready kernel-experience summary (inline).
allowed-tools: Read, Write, Edit, Bash(python3 __SKILL_DIR__/scripts/session_ctl.py:*), Bash(bash __SKILL_DIR__/scripts/finalize_extraction.sh:*)
---
Finish kernel-experience capture for the current session: stop monitoring, then
distill the remaining slice into structured records + a human-review summary,
**inline** so the user can review it now.

Step 0 — mechanical prep (disarms this session, freezes the end, and writes the
filtered final slice + workspace digest). Read its JSON header carefully:

!`python3 __SKILL_DIR__/scripts/session_ctl.py prep-done --cwd "$(pwd)"`

The JSON gives you: `pending_path`, `operator` (may be null), `mode`,
`session_id`, `freeze_uuid`, `mem_version`, `cwd`, `new_turns`, `skill_dir`,
`guides` (array of extraction-guide files), `summaries` (array of summary-skeleton
files).

Now do the following, in order:

1. **Read the inputs**: the file at `pending_path` (the already-filtered slice —
   do NOT re-parse raw jsonl), then **every file listed in the JSON's `guides`
   array** (they live under `templates/extraction/` and define what you MAY / MUST
   / must NOT write, and HOW) and, for schema + controlled tags,
   `__SKILL_DIR__/SKILL.md`.

2. **Resolve the operator**: use `operator` from the JSON if non-null; otherwise
   infer it from the slice (the kernel actually being optimized) and slugify it
   (lowercase, underscores). Call it `<op>`. All output goes under
   `__SKILL_DIR__/knowledge/<op>/`.

3. **Write records**: for each attempt worth remembering (per the guide), write one
   `__SKILL_DIR__/knowledge/<op>/<slug>.json` following the SKILL.md schema —
   numbers-first, both positive and negative, grounded attribution, controlled
   tags. If the background worker had earlier filed anything under
   `knowledge/_inbox/` for this operator, move it into `knowledge/<op>/`.

4. **Consolidate**: merge near-duplicate attempts (same method across adjacent
   versions) into trajectory records with numbers (e.g. v0 2399us → v2 100us → v4
   47us), keep every distinct pitfall, delete merged-away files.

5. **Write the review summary**: pick the best-fit skeleton from the JSON's
   `summaries` array (they live under `templates/summary/`; if there's only one,
   use it), fill it in, and save it as
   `__SKILL_DIR__/knowledge/<op>/SESSION_<session_id first 8 chars>_<YYYY-MM-DD>.md`.

6. **Finalize (global-locked merge + checkpoint + commit)** — run exactly once with
   the values from the JSON header:

   `bash __SKILL_DIR__/scripts/finalize_extraction.sh <session_id> <freeze_uuid> <mem_version>`

7. **Report** to the user: operator, how many records were written/consolidated,
   the path to the review summary, and the manual promotion command (the
   human-review gate — do NOT run it yourself):

   `bash __SKILL_DIR__/scripts/promote_wiki.sh <path-to>/atrex-kernel-agent/gpu-wiki --dry-run`

If the JSON shows `new_turns: 0` and there is genuinely nothing new to record,
skip steps 3–5, still run step 6 (to advance the checkpoint), and tell the user
the session was already up to date and is now disarmed.
