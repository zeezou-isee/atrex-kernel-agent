#!/usr/bin/env python3
"""
checkpoint.py — per-session extraction state for the kernel-experience skill.

Keeps track, for each session_id, of:
  - last_uuid:        last transcript uuid already extracted (checkpoint pointer)
  - last_extract_ts:  epoch seconds of the last extraction (for debounce)
  - last_mem_version: highest atrex memory/vN.json version seen (pacing signal)

State lives in state/checkpoint.json (git-tracked). A sibling .lock file gives a
best-effort exclusive lock so two concurrent extractions never interleave.

Subcommands
-----------
  freeze SESSION.jsonl
        Print the current last uuid of the session (the frozen end boundary).
        Used as extract_transcript.py --freeze-at, so anything appended after
        the trigger is left for the next run.

  since SESSION_ID
        Print the stored last_uuid for a session ("" if none) — pass to
        extract_transcript.py --since.

  should-run SESSION_ID [--debounce-min N] [--mem-version V]
        Exit 0 if an extraction should run now, 1 otherwise. Gate =
        (a new memory version V is provided that exceeds last_mem_version)
        AND (now - last_extract_ts >= debounce). With no --mem-version the
        version gate is skipped (offline / manual use) and only debounce applies.

  advance SESSION_ID --last-uuid U [--mem-version V]
        Record that extraction completed up to uuid U at the current time.

  lock / unlock SESSION_ID
        Acquire / release the advisory lock (used around a full extract cycle).

  show [SESSION_ID]
        Dump state (all sessions, or one).

Time note: uses time.time(); callers in restricted sandboxes may pass --now.
"""
from __future__ import annotations

import argparse
import errno
import json
import os
import sys
import time
from pathlib import Path

STATE_DIR = Path(__file__).resolve().parent.parent / "state"
STATE_FILE = STATE_DIR / "checkpoint.json"


def _now(args):
    return float(getattr(args, "now", None) or time.time())


def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def save_state(state):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE_FILE)


def entry(state, sid):
    return state.setdefault(sid, {"last_uuid": None, "last_extract_ts": 0,
                                  "last_mem_version": -1})


def last_uuid_of(session_path: Path):
    """Return the last uuid-bearing record in a .jsonl (the frozen end)."""
    last = None
    with session_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("uuid"):
                last = obj["uuid"]
    return last


def lock_path(sid):
    return STATE_DIR / f".{sid}.lock"


def cmd_freeze(args):
    p = Path(args.session)
    if not p.exists():
        sys.stderr.write(f"session not found: {p}\n")
        return 2
    u = last_uuid_of(p)
    print(u or "")
    return 0


def cmd_since(args):
    st = load_state()
    e = st.get(args.session_id)
    print((e or {}).get("last_uuid") or "")
    return 0


def cmd_memversion(args):
    """Print the last processed atrex memory version (-1 if none)."""
    st = load_state()
    e = st.get(args.session_id) or {}
    print(int(e.get("last_mem_version", -1)))
    return 0


def cmd_should_run(args):
    st = load_state()
    e = entry(st, args.session_id)
    now = _now(args)
    # debounce gate
    if now - float(e.get("last_extract_ts", 0)) < args.debounce_min * 60:
        return 1
    # version gate (only if a version was supplied)
    if args.mem_version is not None:
        if int(args.mem_version) <= int(e.get("last_mem_version", -1)):
            return 1
    return 0


def cmd_advance(args):
    st = load_state()
    e = entry(st, args.session_id)
    e["last_uuid"] = args.last_uuid
    e["last_extract_ts"] = _now(args)
    if args.mem_version is not None:
        e["last_mem_version"] = max(int(e.get("last_mem_version", -1)), int(args.mem_version))
    save_state(st)
    print(json.dumps(e, ensure_ascii=False))
    return 0


def cmd_lock(args):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    lp = lock_path(args.session_id)
    try:
        fd = os.open(str(lp), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError as ex:
        if ex.errno == errno.EEXIST:
            # stale-lock guard: break locks older than 2h
            try:
                if time.time() - lp.stat().st_mtime > 2 * 3600:
                    lp.unlink()
                    fd = os.open(str(lp), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                else:
                    sys.stderr.write("locked\n")
                    return 1
            except OSError:
                sys.stderr.write("locked\n")
                return 1
        else:
            raise
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return 0


def cmd_unlock(args):
    lp = lock_path(args.session_id)
    try:
        lp.unlink()
    except FileNotFoundError:
        pass
    return 0


def cmd_show(args):
    st = load_state()
    if args.session_id:
        print(json.dumps(st.get(args.session_id, {}), indent=2, ensure_ascii=False))
    else:
        print(json.dumps(st, indent=2, ensure_ascii=False))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("freeze"); p.add_argument("session"); p.set_defaults(fn=cmd_freeze)
    p = sub.add_parser("since"); p.add_argument("session_id"); p.set_defaults(fn=cmd_since)
    p = sub.add_parser("memversion"); p.add_argument("session_id"); p.set_defaults(fn=cmd_memversion)

    p = sub.add_parser("should-run")
    p.add_argument("session_id")
    p.add_argument("--debounce-min", type=float, default=40.0)
    p.add_argument("--mem-version", type=int, default=None)
    p.add_argument("--now", type=float, default=None)
    p.set_defaults(fn=cmd_should_run)

    p = sub.add_parser("advance")
    p.add_argument("session_id")
    p.add_argument("--last-uuid", required=True)
    p.add_argument("--mem-version", type=int, default=None)
    p.add_argument("--now", type=float, default=None)
    p.set_defaults(fn=cmd_advance)

    p = sub.add_parser("lock"); p.add_argument("session_id"); p.set_defaults(fn=cmd_lock)
    p = sub.add_parser("unlock"); p.add_argument("session_id"); p.set_defaults(fn=cmd_unlock)
    p = sub.add_parser("show"); p.add_argument("session_id", nargs="?"); p.set_defaults(fn=cmd_show)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
