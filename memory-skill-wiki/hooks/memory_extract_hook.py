#!/usr/bin/env python3
"""
memory_extract_hook.py — non-blocking Stop / SubagentStop hook for atrex runs.

Registered in the atrex Claude settings.json alongside atrex's own hooks. On each
Stop it decides — cheaply — whether a *new* optimization milestone has landed, and
if so launches a fully detached background worker (run_extraction.sh) that distills
kernel-optimization experience from the live session. It then exits 0 immediately.

CONTRACT (do not violate — atrex's own Stop hook drives the optimization loop):
  * Never print a JSON decision / never block / never inject into the main agent.
  * Always exit 0, even on error. This hook is a silent side-effect only.

Gate (keeps frequency low, updates silent, works for AKA AND vibe coding):
  * armed       = state/armed/<sid>.json exists (written by /arm-run). Checked
                  FIRST and cheaply, so every unrelated session is a pure no-op.
  * debounce    = >= debounce_min minutes since the last extraction (from the
                  marker; default MEMORY_SKILL_DEBOUNCE_MIN=20); checkpoint.py should-run.
  * new content = detect_change.py finds ANY of: a new atrex memory/vN.json, perf/
                  PASS-FAIL result lines, a kernel git-commit, or >= min_turns new
                  human turns in the slice since the last checkpoint.

Input: hook payload JSON on stdin, providing at least `transcript_path` and
`cwd` (and usually `session_id`, `stop_hook_active`). See Claude Code hook docs.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT = SKILL_DIR / "scripts" / "checkpoint.py"
DETECT = SKILL_DIR / "scripts" / "detect_change.py"
RUNNER = SKILL_DIR / "scripts" / "run_extraction.sh"
DEBOUNCE_MIN = os.environ.get("MEMORY_SKILL_DEBOUNCE_MIN", "20")


def read_payload():
    try:
        raw = sys.stdin.read().strip()
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


def newest_mem_version(roots):
    """Highest N across kernel_opt_*/memory/v<N>.json under any root dir."""
    best = -1
    seen = set()
    for root in roots:
        if not root:
            continue
        rp = Path(root)
        if not rp.is_dir() or str(rp) in seen:
            continue
        seen.add(str(rp))
        for mem_json in rp.glob("kernel_opt_*/memory/v*.json"):
            m = re.search(r"v(\d+)\.json$", mem_json.name)
            if not m:
                continue
            try:
                if json.loads(mem_json.read_text(encoding="utf-8")).get("masked") is True:
                    continue
            except Exception:
                pass
            best = max(best, int(m.group(1)))
    return best


def session_id_from(payload, transcript_path):
    sid = payload.get("session_id") or payload.get("sessionId")
    if sid:
        return sid
    if transcript_path:
        name = Path(transcript_path).name
        return name[:-6] if name.endswith(".jsonl") else Path(transcript_path).stem
    return None


def main():
    payload = read_payload()
    transcript_path = payload.get("transcript_path") or payload.get("transcriptPath")
    cwd = payload.get("cwd") or os.getcwd()
    sid = session_id_from(payload, transcript_path)

    # Missing essentials -> silently do nothing.
    if not transcript_path or not Path(transcript_path).exists() or not sid:
        return 0

    # GATE 0 (cheapest, FIRST): only ARMED sessions are monitored. This single
    # stat() is what keeps the global hook a pure no-op for every unrelated
    # session — nothing else runs unless /arm-run wrote this marker.
    marker_file = SKILL_DIR / "state" / "armed" / f"{sid}.json"
    if not marker_file.exists():
        return 0
    try:
        marker = json.loads(marker_file.read_text(encoding="utf-8"))
    except Exception:
        return 0

    cwd = marker.get("cwd") or cwd
    debounce = marker.get("debounce_min", DEBOUNCE_MIN)
    min_turns = marker.get("min_turns", 4)
    operator = marker.get("operator") or ""

    def ckpt(*a):
        return subprocess.run([sys.executable, str(CHECKPOINT), *a],
                              capture_output=True, text=True)

    # GATE 1: debounce only (time). Content is judged separately (GATE 2) so that
    # vibe-coding sessions with no memory/vN.json are covered too.
    if ckpt("should-run", sid, "--debounce-min", str(debounce)).returncode != 0:
        return 0

    # Slice bounds for the content gate / the worker.
    since = ckpt("since", sid).stdout.strip()
    freeze = ckpt("freeze", transcript_path).stdout.strip()
    prior_mv = ckpt("memversion", sid).stdout.strip() or "-1"

    # GATE 2: meaningful new content (any of: new memory version / result lines /
    # kernel commit / >= min_turns new human turns).
    detect_cmd = [sys.executable, str(DETECT), "--session", transcript_path,
                  "--cwd", cwd, "--prior-mem-version", str(prior_mv),
                  "--min-turns", str(min_turns)]
    if since:
        detect_cmd += ["--since", since]
    if freeze:
        detect_cmd += ["--freeze", freeze]
    if subprocess.run(detect_cmd, capture_output=True).returncode != 0:
        return 0

    mem_version = newest_mem_version([cwd])  # runner/checkpoint bookkeeping (-1 ok)

    # Launch fully detached background worker; return immediately. Rely on
    # start_new_session=True for the detach (portable; macOS has no `setsid`).
    logf = SKILL_DIR / "state" / "extraction.log"
    logf.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(logf, "a") as lf:
            subprocess.Popen(
                ["bash", str(RUNNER), transcript_path, sid, freeze,
                 str(mem_version), cwd, operator],
                stdin=subprocess.DEVNULL, stdout=lf, stderr=lf,
                start_new_session=True, cwd=str(SKILL_DIR),
            )
    except Exception:
        # Never let a spawn failure surface to the main agent.
        pass
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        # Absolute backstop: this hook must never fail loudly.
        sys.exit(0)
