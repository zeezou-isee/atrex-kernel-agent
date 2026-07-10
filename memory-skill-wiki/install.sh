#!/usr/bin/env bash
# install.sh — one-click install of the kernel-experience-memory skill, its three
# slash commands (/arm-run, /done-run, /run-status), and the non-blocking
# Stop/SubagentStop monitoring hook.
#
# By DEFAULT this installs all three (skill + commands + hook). The hook is a pure
# no-op in any session that has NOT been armed with /arm-run, so a global install
# is safe. Pass --without-hook to install only the skill + commands.
#
# Usage:
#   ./install.sh                     # skill + commands + hook (default)
#   ./install.sh --global            # force target ~/.claude (recommended)
#   ./install.sh --without-hook      # skill + commands only (no monitoring hook)
#   ./install.sh --with-hook         # explicit: also install the hook (default)
#   ./install.sh --claude-dir DIR    # target a specific .claude dir
#   ./install.sh --debounce-min N    # hook debounce minutes (default 20)
#   ./install.sh --uninstall         # remove skill link, commands, and hook
#   ./install.sh --help
#
# The skill is symlinked into <.claude>/skills/ (so knowledge/ wiki/ state/ keep
# accumulating in this source dir); the commands are COPIED into
# <.claude>/commands/ with the skill's absolute path baked in (re-run to update).
# Target .claude is auto-detected by walking up from the cwd (falls back to
# ~/.claude); --global forces ~/.claude.
set -euo pipefail

SKILL_NAME="kernel-experience-memory"
HOOK_TAG="kernel-experience-memory-hook-v1"
COMMANDS="arm-run done-run run-status"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WITH_HOOK=1            # default ON — commands are useless without the hook
DEBOUNCE=""
CLAUDE_DIR=""
MODE="install"

c_g(){ printf '\033[32m%s\033[0m\n' "$*"; }
c_y(){ printf '\033[33m%s\033[0m\n' "$*"; }
c_r(){ printf '\033[31m%s\033[0m\n' "$*" >&2; }
die(){ c_r "error: $*"; exit 1; }

find_claude_dir() {
  local d="$PWD"
  while [ "$d" != "/" ]; do
    [ -d "$d/.claude" ] && { echo "$d/.claude"; return; }
    d="$(dirname "$d")"
  done
  echo "$HOME/.claude"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --with-hook)      WITH_HOOK=1; shift ;;
    --without-hook)   WITH_HOOK=0; shift ;;
    --global)         CLAUDE_DIR="$HOME/.claude"; shift ;;
    --debounce-min)   DEBOUNCE="${2:?}"; WITH_HOOK=1; shift 2 ;;
    --claude-dir)     CLAUDE_DIR="${2:?}"; shift 2 ;;
    --uninstall)      MODE="uninstall"; shift ;;
    -h|--help)        sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)                die "unknown option: $1 (see --help)" ;;
  esac
done

[ -z "$CLAUDE_DIR" ] && CLAUDE_DIR="$(find_claude_dir)"
SKILLS_DIR="$CLAUDE_DIR/skills"
COMMANDS_DIR="$CLAUDE_DIR/commands"
SETTINGS="$CLAUDE_DIR/settings.json"
SKILL_LINK="$SKILLS_DIR/$SKILL_NAME"
HOOK_SCRIPT="$SRC/hooks/memory_extract_hook.py"

command -v git >/dev/null 2>&1 || die "git is required"

# ---------------------------------------------------------------- uninstall ---
if [ "$MODE" = "uninstall" ]; then
  if [ -L "$SKILL_LINK" ]; then rm -f "$SKILL_LINK"; c_g "removed skill link: $SKILL_LINK"
  else c_y "no skill link at $SKILL_LINK (skipped)"; fi

  for c in $COMMANDS; do
    f="$COMMANDS_DIR/$c.md"
    # only remove commands that are ours (reference this skill's scripts)
    if [ -f "$f" ] && grep -q "session_ctl.py" "$f" 2>/dev/null; then
      rm -f "$f"; c_g "removed command: $f"
    fi
  done

  if [ -f "$SETTINGS" ] && command -v jq >/dev/null 2>&1; then
    tmp="$(mktemp)"
    jq --arg tag "$HOOK_TAG" '
      (.hooks.Stop         |= (if . then map(select(._tag != $tag)) else . end))
      | (.hooks.SubagentStop |= (if . then map(select(._tag != $tag)) else . end))
      | (if (.hooks.Stop // [])         == [] then del(.hooks.Stop)         else . end)
      | (if (.hooks.SubagentStop // []) == [] then del(.hooks.SubagentStop) else . end)
    ' "$SETTINGS" > "$tmp" && mv "$tmp" "$SETTINGS"
    c_g "removed hook entries (tag $HOOK_TAG) from $SETTINGS"
  fi
  c_g "uninstall complete."
  exit 0
fi

# ------------------------------------------------------- prepare the source ---
chmod +x "$SRC"/scripts/*.py "$SRC"/scripts/*.sh "$SRC"/hooks/*.py 2>/dev/null || true
for d in knowledge state; do
  if [ ! -d "$SRC/$d/.git" ]; then
    git init -q "$SRC/$d"
    git -C "$SRC/$d" config user.email "memory-skill@local"
    git -C "$SRC/$d" config user.name  "memory-skill"
    c_g "initialized git repo: $d/"
  fi
done

# ---------------------------------------------------------- register skill ----
mkdir -p "$SKILLS_DIR"
if [ -e "$SKILL_LINK" ] && [ ! -L "$SKILL_LINK" ]; then
  die "$SKILL_LINK exists and is not a symlink; move it aside first"
fi
ln -sfn "$SRC" "$SKILL_LINK"
c_g "registered skill '$SKILL_NAME' -> $SKILL_LINK"

# --------------------------------------------------------- install commands ---
mkdir -p "$COMMANDS_DIR"
for c in $COMMANDS; do
  src_cmd="$SRC/commands/$c.md"
  [ -f "$src_cmd" ] || { c_y "missing command source: $src_cmd (skipped)"; continue; }
  if [ -e "$COMMANDS_DIR/$c.md" ] && ! grep -q "session_ctl.py" "$COMMANDS_DIR/$c.md" 2>/dev/null; then
    die "$COMMANDS_DIR/$c.md exists and is not ours; move it aside first"
  fi
  sed "s|__SKILL_DIR__|$SRC|g" "$src_cmd" > "$COMMANDS_DIR/$c.md"
  c_g "installed command: /$c"
done

# ------------------------------------------------------------ install hook ----
if [ "$WITH_HOOK" = "1" ]; then
  command -v jq >/dev/null 2>&1 || die "jq is required to install the hook"
  HOOK_CMD="python3 $HOOK_SCRIPT"
  [ -n "$DEBOUNCE" ] && HOOK_CMD="MEMORY_SKILL_DEBOUNCE_MIN=$DEBOUNCE $HOOK_CMD"
  [ -f "$SETTINGS" ] || echo '{}' > "$SETTINGS"
  tmp="$(mktemp)"
  jq --arg cmd "$HOOK_CMD" --arg tag "$HOOK_TAG" '
    .hooks = (.hooks // {})
    | .hooks.Stop = (((.hooks.Stop // []) | map(select(._tag != $tag)))
        + [{_tag:$tag, hooks:[{type:"command", command:$cmd}]}])
    | .hooks.SubagentStop = (((.hooks.SubagentStop // []) | map(select(._tag != $tag)))
        + [{_tag:$tag, hooks:[{type:"command", command:$cmd}]}])
  ' "$SETTINGS" > "$tmp" && mv "$tmp" "$SETTINGS"
  c_g "registered Stop + SubagentStop hook in $SETTINGS"
  [ -n "$DEBOUNCE" ] && c_g "  debounce = ${DEBOUNCE} min"
else
  c_y "hook NOT installed (--without-hook). /arm-run will warn until it is installed."
fi

# --------------------------------------------------------------- summary ------
echo
c_g "=== install complete ==="
echo "  skill dir : $SKILL_LINK  ->  $SRC"
echo "  commands  : $COMMANDS_DIR/{arm-run,done-run,run-status}.md"
echo "  settings  : $SETTINGS"
echo "  use       : /arm-run [operator]  ·  /run-status  ·  /done-run"
[ "$WITH_HOOK" = "1" ] && echo "  monitoring: fires on Stop/SubagentStop, only for sessions armed via /arm-run"
echo
c_y "Restart the coding runtime (or open a new session) so the skill/commands/hook load."
