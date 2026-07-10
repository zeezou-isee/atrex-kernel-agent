#!/usr/bin/env python3
"""
locate_session.py — find the Claude Code session transcript for a directory.

Claude Code stores transcripts under ~/.claude/projects/<mangled-cwd>/<id>.jsonl
where <mangled-cwd> is the absolute cwd with every non-alphanumeric char replaced
by '-'. This resolves that path so the OFFLINE entry can say "extract from my
current session" without knowing the id, and so the hook has a fallback when the
payload omits transcript_path.

Usage:
    locate_session.py [--cwd DIR] [--all] [--projects-base DIR]
      default: print the most-recently-modified .jsonl for --cwd (default $PWD)
      --all:   print every .jsonl (newest first), one per line
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


def mangle(cwd: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "-", cwd)


def project_dir(cwd: str, base: Path) -> Path:
    return base / mangle(os.path.abspath(cwd))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cwd", default=os.getcwd())
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--projects-base", type=Path,
                    default=Path(os.path.expanduser("~/.claude/projects")))
    args = ap.parse_args(argv)

    pdir = project_dir(args.cwd, args.projects_base)
    if not pdir.is_dir():
        sys.stderr.write(f"no project dir for cwd={args.cwd} at {pdir}\n")
        return 1
    sessions = sorted(pdir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not sessions:
        sys.stderr.write(f"no .jsonl transcripts under {pdir}\n")
        return 1
    if args.all:
        for p in sessions:
            print(p)
    else:
        print(sessions[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
