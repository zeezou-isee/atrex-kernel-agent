#!/usr/bin/env python3
"""
detect_change.py — does the new transcript slice hold meaningful new content?

hooks/memory_extract_hook.py calls this for an ARMED session (after the debounce
gate) to decide whether launching an extraction is worthwhile. Exit 0 (+ a short
reason on stdout) if ANY of four signals fires, else exit 1:

  1. version : a new un-masked kernel_opt_*/memory/v<N>.json  (> --prior-mem-version)
  2. result  : perf / PASS-FAIL / error lines in the slice   (RESULT_SIGNAL_RE)
  3. commit  : a git-commit / memory_manager submit in the slice's Bash tool_use
  4. turns   : >= --min-turns new human turns in the slice

Reuses extract_transcript.py (load_jsonl, slice_by_uuid, human_text, blocks,
RESULT_SIGNAL_RE) so the "result signal" definition stays identical to Step 1.

Usage:
    detect_change.py --session S.jsonl [--since U] [--freeze U] [--cwd DIR]
        [--prior-mem-version N] [--min-turns N]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from extract_transcript import (load_jsonl, slice_by_uuid, human_text, blocks,
                                RESULT_SIGNAL_RE)

# git commit (in any form: `git commit`, `git -C x commit`, `... && git commit`)
# or AKA's memory_manager submit.
COMMIT_RE = re.compile(r"(\bgit\b[^\n]{0,80}\bcommit\b|memory_manager\.py)", re.I)


def newest_mem_version(cwd):
    best = -1
    root = Path(cwd) if cwd else None
    if not root or not root.is_dir():
        return best
    workspaces = list(root.glob("kernel_opt_*"))
    if (root / "memory").is_dir():
        workspaces.append(root)
    for ws in workspaces:
        md = ws / "memory"
        if not md.is_dir():
            continue
        for mem in md.glob("v*.json"):
            m = re.search(r"v(\d+)\.json$", mem.name)
            if not m:
                continue
            try:
                if json.loads(mem.read_text(encoding="utf-8")).get("masked") is True:
                    continue
            except Exception:
                pass
            best = max(best, int(m.group(1)))
    return best


def _text_of(block):
    if not isinstance(block, dict):
        return ""
    t = block.get("type")
    if t == "text":
        return block.get("text") or ""
    if t == "tool_result":
        c = block.get("content")
        if isinstance(c, list):
            return "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
        return str(c or "")
    return ""


def scan(records):
    """(new_human_turns, result_hit|None, commit_hit|None) over the slice."""
    new_turns = 0
    result_hit = None
    commit_hit = None
    for r in records:
        if human_text(r) is not None:
            new_turns += 1
        for b in blocks(r.get("message", {})):
            bt = b.get("type")
            if bt in ("text", "tool_result") and result_hit is None:
                for ln in _text_of(b).splitlines():
                    ln = ln.strip()
                    if ln and len(ln) <= 400 and not re.match(r"^\d+\t", ln) \
                            and RESULT_SIGNAL_RE.search(ln):
                        result_hit = ln[:120]
                        break
            elif bt == "tool_use" and commit_hit is None and b.get("name") == "Bash":
                cmd = (b.get("input") or {}).get("command") or ""
                if COMMIT_RE.search(cmd):
                    commit_hit = (cmd.strip().splitlines() or [""])[0][:120]
    return new_turns, result_hit, commit_hit


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True, type=Path)
    ap.add_argument("--since", default=None)
    ap.add_argument("--freeze", default=None)
    ap.add_argument("--cwd", default=None)
    ap.add_argument("--prior-mem-version", type=int, default=-1)
    ap.add_argument("--min-turns", type=int, default=4)
    args = ap.parse_args(argv)

    if not args.session.exists():
        sys.stderr.write(f"session not found: {args.session}\n")
        return 1

    records = slice_by_uuid(load_jsonl(args.session), args.since or None, args.freeze or None)
    new_turns, result_hit, commit_hit = scan(records)
    mv = newest_mem_version(args.cwd) if args.cwd else -1

    reasons = []
    if mv > args.prior_mem_version:
        reasons.append(f"mem v{mv}")
    if result_hit:
        reasons.append(f"result[{result_hit}]")
    if commit_hit:
        reasons.append(f"commit[{commit_hit}]")
    if new_turns >= args.min_turns:
        reasons.append(f"{new_turns} new turns")

    if reasons:
        print("TRIGGER: " + " | ".join(reasons))
        return 0
    print(f"no-trigger (turns={new_turns} < {args.min_turns}, "
          f"mem={mv} <= {args.prior_mem_version}, no result/commit)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
