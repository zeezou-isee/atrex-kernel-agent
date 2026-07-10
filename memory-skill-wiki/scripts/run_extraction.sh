#!/usr/bin/env bash
# run_extraction.sh — the detached worker for real-time experience extraction.
#
# Invoked (fully detached) by hooks/memory_extract_hook.py. Runs the two-step
# pipeline on the frozen slice of a live session, writes structured records,
# regenerates the wiki, advances the checkpoint, and commits — all in an isolated
# process that never touches the main agent's context.
#
# Args:
#   $1 SESSION_JSONL   absolute path to the main session transcript
#   $2 SESSION_ID      session id (checkpoint key)
#   $3 FREEZE_UUID     frozen end boundary (from checkpoint.py freeze)
#   $4 MEM_VERSION     highest atrex memory/vN.json version seen (pacing signal)
#   $5 CWD             session cwd (for the workspace digest)
#   $6 OPERATOR        operator slug -> records land in knowledge/<op>/ (else _inbox)
#
# LLM step: uses $MEMORY_SKILL_LLM_CMD if set (receives the filtered-md path as
# its last arg; must write knowledge/*.json), else falls back to `claude -p`.
# If neither produces new records, the checkpoint is NOT advanced and the
# filtered slice is kept under state/pending/ for later processing.
set -uo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SKILL_DIR" || exit 0

SESSION_JSONL="${1:?session jsonl}"; SESSION_ID="${2:?session id}"
FREEZE_UUID="${3:-}"; MEM_VERSION="${4:--1}"; CWD="${5:-}"; OPERATOR="${6:-}"
DEBOUNCE_MIN="${MEMORY_SKILL_DEBOUNCE_MIN:-20}"

# Per-operator knowledge dir (records land here; the LLM is told this path).
OP_SLUG="$(printf '%s' "$OPERATOR" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | tr -cd 'a-z0-9._-')"
[ -z "$OP_SLUG" ] && OP_SLUG="_inbox"
KDIR="knowledge/${OP_SLUG}"
mkdir -p "$KDIR"

log() { printf '[%s] %s\n' "$(date -u +%FT%TZ 2>/dev/null || echo now)" "$*"; }

# --- lock (best effort; bail if another worker holds it) ---
if ! python3 scripts/checkpoint.py lock "$SESSION_ID" 2>/dev/null; then
  log "another extraction holds the lock for $SESSION_ID; exiting"
  exit 0
fi
trap 'python3 scripts/checkpoint.py unlock "$SESSION_ID" 2>/dev/null' EXIT

mkdir -p state/pending
SINCE="$(python3 scripts/checkpoint.py since "$SESSION_ID")"

# Nothing appended since last extraction -> nothing to do (avoids empty re-runs).
if [ -n "$FREEZE_UUID" ] && [ "$SINCE" = "$FREEZE_UUID" ]; then
  log "no new content since last checkpoint ($SINCE); skipping"
  exit 0
fi

PENDING="state/pending/${SESSION_ID}_${FREEZE_UUID:-end}.md"

# --- Step 1: deterministic filter/segment (frozen slice only) ---
if ! python3 scripts/extract_transcript.py "$SESSION_JSONL" \
      ${SINCE:+--since "$SINCE"} ${FREEZE_UUID:+--freeze-at "$FREEZE_UUID"} \
      --out "$PENDING" 2>>state/extraction.log; then
  log "extract_transcript failed"; exit 0
fi
log "filtered slice -> $PENDING ($(wc -l <"$PENDING") lines)"

# --- Step 1b: attach atrex workspace ground-truth numbers (authoritative) ---
# Numbers live in kernel_opt_*/memory/v*.json, not the transcript. Only digest
# versions newer than what we already processed.
PRIOR_MV="$(python3 scripts/checkpoint.py memversion "$SESSION_ID")"
if [ -n "$CWD" ] && [ -d "$CWD" ]; then
  {
    echo; echo "---";
    python3 scripts/collect_workspace.py --cwd "$CWD" --since-version "${PRIOR_MV:--1}"
  } >> "$PENDING" 2>>state/extraction.log
  log "appended workspace digest (versions > ${PRIOR_MV:--1})"
fi


# --- Step 2: LLM distillation into knowledge/<op>/*.json ---
# Auto-scan the extraction-guide dir (drop-in: any .md there is used; README and
# _/.-prefixed files skipped). Add a guide by dropping a file in templates/extraction/.
GUIDES=""
for gf in templates/extraction/*.md; do
  [ -e "$gf" ] || continue
  # ignore any README* (incl. README.zh.md) and _/.-prefixed files
  case "$(basename "$gf")" in _*|.*|[Rr][Ee][Aa][Dd][Mm][Ee]*) continue;; esac
  GUIDES="${GUIDES}${SKILL_DIR}/${gf} "
done
[ -z "$GUIDES" ] && GUIDES="(none; use SKILL.md) "
before="$(find knowledge -name '*.json' -not -path '*/.git/*' 2>/dev/null | wc -l | tr -d ' ')"
PROMPT="Use the kernel-experience-memory skill. Read the already-filtered transcript at ${SKILL_DIR}/${PENDING}. It is one session's new slice. Read ALL of these extraction guide file(s): ${GUIDES}(each explains what you MAY write / MUST write / must NOT write / HOW to write) plus ${SKILL_DIR}/SKILL.md (record schema, controlled tags, numbers-first hard rules). Then extract each optimization attempt worth remembering and write ONE record per attempt as ${SKILL_DIR}/${KDIR}/<slug>.json (operator '${OPERATOR:-unknown}'). If the operator is unknown, infer it from the slice and write to ${SKILL_DIR}/knowledge/<inferred_operator>/<slug>.json instead. Do not run merge_wiki or advance the checkpoint; the runner does that. Be silent."

if [ -n "${MEMORY_SKILL_LLM_CMD:-}" ]; then
  log "LLM step via MEMORY_SKILL_LLM_CMD"
  # shellcheck disable=SC2086
  MEMORY_SKILL_PENDING="$PENDING" bash -c "$MEMORY_SKILL_LLM_CMD \"$PENDING\"" >>state/extraction.log 2>&1
  llm_rc=$?
elif command -v claude >/dev/null 2>&1; then
  log "LLM step via claude -p (headless)"
  claude -p "$PROMPT" --permission-mode acceptEdits >>state/extraction.log 2>&1
  llm_rc=$?
else
  log "no LLM available; leaving $PENDING for later, NOT advancing checkpoint"
  exit 0
fi
after="$(find knowledge -name '*.json' -not -path '*/.git/*' 2>/dev/null | wc -l | tr -d ' ')"
log "LLM step rc=$llm_rc; records $before -> $after"

if [ "$after" -le "$before" ]; then
  log "no new records produced; keeping pending slice, NOT advancing checkpoint"
  exit 0
fi

# --- git-version the processed slice (user asked: git-manage the session) ---
# Append the normalized+digested slice to a per-session history file, so the git
# diff between commits == exactly the new content processed this trigger. Done
# only now (records succeeded) so a failed LLM step never desyncs history.
mkdir -p state/sessions
{ echo; echo "<!-- === extract @ ${FREEZE_UUID:-end} (mem v$MEM_VERSION) === -->"; cat "$PENDING"; } \
  >> "state/sessions/${SESSION_ID}.md"
rm -f "$PENDING"

# --- merge wiki + advance checkpoint + commit (global-locked, shared tail) ---
# Serialized across concurrent armed sessions so checkpoint.json / wiki / git
# never race; the slow LLM step above already ran unlocked and in parallel.
bash scripts/finalize_extraction.sh "$SESSION_ID" "$FREEZE_UUID" "$MEM_VERSION"

log "extraction complete for $SESSION_ID"
exit 0
