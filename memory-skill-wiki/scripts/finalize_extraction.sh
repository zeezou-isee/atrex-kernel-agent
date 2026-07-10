#!/usr/bin/env bash
# finalize_extraction.sh — global-locked write-back tail shared by
# run_extraction.sh (background worker) and the /done-run command.
#
#   $1 SESSION_ID     checkpoint key
#   $2 FREEZE_UUID    frozen end boundary to advance the checkpoint to (may be empty)
#   $3 MEM_VERSION    highest atrex memory/vN.json version seen (may be -1)
#
# Under a cross-session GLOBAL lock (checkpoint.py lock __global__ — macOS has no
# flock): regenerate the wiki, advance the checkpoint, and commit the knowledge/ +
# state/ repos (and wiki/ if it is a git repo). The slow LLM/extraction step has
# already run; this tail is ~1-2s, so serializing it across concurrent armed
# sessions is cheap and prevents checkpoint.json / wiki / git races.
set -uo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SKILL_DIR" || exit 0

SID="${1:?session id}"; FREEZE="${2:-}"; MV="${3:--1}"

log() { printf '[%s] %s\n' "$(date -u +%FT%TZ 2>/dev/null || echo now)" "$*"; }

acquire_global() {
  local i
  for i in $(seq 1 90); do          # ~180s max wait (tail is ~1-2s, so ample)
    if python3 scripts/checkpoint.py lock __global__ 2>/dev/null; then return 0; fi
    sleep 2
  done
  return 1
}

if ! acquire_global; then
  log "could not acquire global lock (~180s) for $SID; skipping write-back"
  exit 0
fi
trap 'python3 scripts/checkpoint.py unlock __global__ 2>/dev/null' EXIT

python3 scripts/merge_wiki.py >>state/extraction.log 2>&1 || log "merge_wiki failed"

if [ -n "$FREEZE" ]; then
  python3 scripts/checkpoint.py advance "$SID" --last-uuid "$FREEZE" --mem-version "$MV" \
      >>state/extraction.log 2>&1 || log "checkpoint advance failed"
fi

for repo in knowledge state; do
  if [ -d "$repo/.git" ]; then
    git -C "$repo" add -A 2>/dev/null
    git -C "$repo" commit -q -m "extract: $SID @ ${FREEZE:-end} (mem v$MV)" 2>/dev/null || true
  fi
done
git -C wiki rev-parse --git-dir >/dev/null 2>&1 && {
  git -C wiki add -A 2>/dev/null
  git -C wiki commit -q -m "wiki: $SID" 2>/dev/null || true
}

log "finalize complete for $SID @ ${FREEZE:-end} (mem v$MV)"
exit 0
