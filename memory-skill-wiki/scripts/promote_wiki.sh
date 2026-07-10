#!/usr/bin/env bash
# promote_wiki.sh — copy the reviewed local wiki/ into a target gpu-wiki.
#
# This is the MANUAL, human-review-gated step: /done-run only regenerates the
# LOCAL wiki/ under this skill. After you have reviewed knowledge/<op>/ and the
# session summary, run this to stage the docs into atrex-kernel-agent/gpu-wiki,
# then review the git diff THERE before committing.
#
# Usage:
#   promote_wiki.sh <gpu-wiki-dir> [--dry-run]
#
# Example:
#   scripts/promote_wiki.sh ../atrex-kernel-agent/gpu-wiki --dry-run
#   scripts/promote_wiki.sh ../atrex-kernel-agent/gpu-wiki
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:?usage: promote_wiki.sh <gpu-wiki-dir> [--dry-run]}"
DRY="${2:-}"

[ -d "$TARGET/docs" ] || { echo "error: $TARGET/docs not found — is that a gpu-wiki dir?" >&2; exit 1; }
SRC="$SKILL_DIR/wiki/docs"
[ -d "$SRC" ] || { echo "error: no local wiki at $SRC — run scripts/merge_wiki.py first" >&2; exit 1; }

opts="-av"
[ "$DRY" = "--dry-run" ] && opts="-avn"

for sub in kernel-opt pitfalls ref-docs; do
  [ -d "$SRC/$sub" ] || continue
  mkdir -p "$TARGET/docs/$sub"
  if command -v rsync >/dev/null 2>&1; then
    rsync $opts "$SRC/$sub/" "$TARGET/docs/$sub/"
  else
    [ "$DRY" = "--dry-run" ] && { echo "(dry-run, no rsync) would copy $SRC/$sub/ -> $TARGET/docs/$sub/"; continue; }
    cp -R "$SRC/$sub/." "$TARGET/docs/$sub/"
  fi
done

echo
echo "done. Review the diff in the target before committing:"
echo "  git -C \"$TARGET\" status && git -C \"$TARGET\" diff"
