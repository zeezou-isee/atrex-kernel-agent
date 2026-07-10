#!/usr/bin/env python3
"""
session_ctl.py — control plane for kernel-experience monitoring (arm/disarm/status).

Backs the /arm-run, /done-run, /run-status slash commands. Resolves the current
Claude Code session for a cwd, writes/removes the "armed" marker that
hooks/memory_extract_hook.py checks, reports status, and prepares the final
done-run slice. Reuses locate_session.py (session resolution), checkpoint.py
(progress state + locks), and extract_transcript.py (slicing helpers). Only ever
writes under state/ (never the transcript, never the AKA workspace).

Subcommands
-----------
  resolve   --cwd DIR
        Print "<session_id>\\t<transcript_path>" for the newest transcript of DIR.

  arm       --cwd DIR [--op OP] [--debounce-min N] [--min-turns N]
        Detect mode (aka|vibe) + operator, write state/armed/<sid>.json, warn if
        the Stop hook is not installed. Human-readable summary.

  disarm    --cwd DIR
        Remove the armed marker (monitoring off).

  status    --cwd DIR [--all]
        Report armed / checkpoint / unprocessed-slice / records / worker / hook.

  prep-done --cwd DIR
        Mechanical half of /done-run: read marker, disarm, freeze, run
        extract_transcript + collect_workspace into state/pending/<sid>_final.md,
        print a one-line JSON header the command body consumes.

Marker schema (state/armed/<session_id>.json):
  {session_id, cwd, transcript_path, operator|null, mode, debounce_min,
   min_turns, armed_at, armed_ts}
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Sibling scripts (same dir -> importable when run as scripts/session_ctl.py).
from locate_session import project_dir
from checkpoint import load_state, lock_path, last_uuid_of, STATE_DIR
from extract_transcript import load_jsonl, slice_by_uuid, human_text

SKILL_DIR = Path(__file__).resolve().parent.parent
SCRIPTS = SKILL_DIR / "scripts"
ARMED_DIR = STATE_DIR / "armed"
PENDING_DIR = STATE_DIR / "pending"
KNOWLEDGE_DIR = SKILL_DIR / "knowledge"
DEFAULT_PROJECTS_BASE = Path(os.path.expanduser("~/.claude/projects"))
DEFAULT_DEBOUNCE_MIN = float(os.environ.get("MEMORY_SKILL_DEBOUNCE_MIN", "20"))
DEFAULT_MIN_TURNS = 4


def slugify(s):
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w.-]+", "-", s)
    return s.strip("-")


def iso(ts):
    try:
        return datetime.fromtimestamp(float(ts), timezone.utc).isoformat(timespec="seconds")
    except Exception:
        return str(ts)


def resolve_session(cwd, base=DEFAULT_PROJECTS_BASE):
    """Newest transcript for cwd -> (session_id, Path) or (None, None)."""
    pdir = project_dir(os.path.abspath(cwd), base)
    if not pdir.is_dir():
        return None, None
    sessions = sorted(pdir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not sessions:
        return None, None
    p = sessions[0]
    sid = p.name[:-6] if p.name.endswith(".jsonl") else p.stem
    return sid, p


def find_kernel_opt(cwd):
    """AKA workspaces (dirs with memory/) under cwd, plus cwd itself if it is one."""
    root = Path(cwd)
    out = []
    if root.is_dir():
        for cand in sorted(root.glob("kernel_opt_*")):
            if (cand / "memory").is_dir():
                out.append(cand)
        if (root / "memory").is_dir():
            out.append(root)
    return out


def newest_mem_version(cwd):
    """Highest un-masked memory/v<N>.json across workspaces under cwd (-1 if none)."""
    best = -1
    for ws in find_kernel_opt(cwd):
        for mem in (ws / "memory").glob("v*.json"):
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


def detect_mode_op(cwd, explicit_op):
    """(mode, operator|None). Explicit arg wins; else kernel_opt_<name> slug; else None."""
    kdirs = find_kernel_opt(cwd)
    mode = "aka" if kdirs else "vibe"
    if explicit_op:
        return mode, (slugify(explicit_op) or None)
    if kdirs:
        name = kdirs[0].name
        if name.startswith("kernel_opt_"):
            name = name[len("kernel_opt_"):]
        return mode, (slugify(name) or None)
    return mode, None


def hook_installed(cwd):
    """True if a Stop/SubagentStop hook referencing memory_extract_hook.py exists."""
    for s in (Path.home() / ".claude" / "settings.json",
              Path(cwd) / ".claude" / "settings.json"):
        try:
            d = json.loads(s.read_text(encoding="utf-8"))
        except Exception:
            continue
        hooks = d.get("hooks", {}) or {}
        for evt in ("Stop", "SubagentStop"):
            for entry in hooks.get(evt, []) or []:
                for h in entry.get("hooks", []) or []:
                    if "memory_extract_hook.py" in (h.get("command", "") or ""):
                        return True
    return False


def marker_path(sid):
    return ARMED_DIR / f"{sid}.json"


def read_marker(sid):
    try:
        return json.loads(marker_path(sid).read_text(encoding="utf-8"))
    except Exception:
        return None


def write_marker(marker):
    ARMED_DIR.mkdir(parents=True, exist_ok=True)
    p = marker_path(marker["session_id"])
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(marker, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def record_count(op):
    def count(d):
        if not d.is_dir():
            return 0
        return sum(1 for x in d.rglob("*.json") if ".git" not in x.parts)
    return count(KNOWLEDGE_DIR / op) if op else count(KNOWLEDGE_DIR)


def list_templates(sub):
    """Active *.md under templates/<sub>/ (skip README + _/.-prefixed files).

    This is the directory-scan that makes templates drop-in: add a .md and it is
    picked up automatically, no code change."""
    d = SKILL_DIR / "templates" / sub
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.md")):
        # ignore any README* (incl. README.zh.md) and _/.-prefixed files
        if p.name.startswith(("_", ".")) or p.name.lower().startswith("readme"):
            continue
        out.append(str(p))
    return out


def checkpoint_entry(sid):
    return (load_state() or {}).get(sid, {}) or {}


def unprocessed_turns(transcript, since):
    """(end_uuid, n_new_human_turns) for the slice after `since`."""
    try:
        recs = list(load_jsonl(Path(transcript)))
    except Exception:
        return None, 0
    end = None
    for r in recs:
        if r.get("uuid"):
            end = r["uuid"]
    sl = slice_by_uuid(recs, since or None, None)
    n = sum(1 for r in sl if human_text(r) is not None)
    return end, n


# --------------------------------------------------------------- commands ----

def cmd_resolve(args):
    sid, tp = resolve_session(args.cwd, args.projects_base)
    if not sid:
        sys.stderr.write(f"no session for cwd={args.cwd}\n")
        return 1
    print(f"{sid}\t{tp}")
    return 0


def cmd_arm(args):
    sid, tp = resolve_session(args.cwd, args.projects_base)
    if not sid:
        sys.stderr.write(f"ERROR: cannot resolve current session for cwd={args.cwd}\n")
        return 2
    mode, op = detect_mode_op(args.cwd, args.op)
    ts = time.time()
    marker = {
        "session_id": sid,
        "cwd": os.path.abspath(args.cwd),
        "transcript_path": str(tp),
        "operator": op,
        "mode": mode,
        "debounce_min": args.debounce_min,
        "min_turns": args.min_turns,
        "armed_at": iso(ts),
        "armed_ts": ts,
    }
    write_marker(marker)
    print(f"ARMED  session={sid[:8]}  mode={mode}  "
          f"operator={op or '(unknown - will infer at summary time)'}")
    print(f"  debounce={args.debounce_min:g}min  min_turns={args.min_turns}  "
          f"-> knowledge/{op or '_inbox'}/")
    print(f"  marker: {marker_path(sid)}")
    if not hook_installed(args.cwd):
        print("")
        print("WARNING: Stop hook not found in settings.json — monitoring will NOT")
        print("fire until it is installed. Run (then restart the session):")
        print(f"  bash {SKILL_DIR}/install.sh --global")
    return 0


def cmd_disarm(args):
    sid, _ = resolve_session(args.cwd, args.projects_base)
    if not sid:
        sys.stderr.write(f"no session for cwd={args.cwd}\n")
        return 1
    m = read_marker(sid)
    p = marker_path(sid)
    if p.exists():
        p.unlink()
        print(f"DISARMED session={sid[:8]}  operator={(m or {}).get('operator')}")
    else:
        print(f"(not armed) session={sid[:8]}")
    return 0


def _status_one(sid, tp=None, cwd=None):
    m = read_marker(sid)
    e = checkpoint_entry(sid)
    op = (m or {}).get("operator")
    L = [f"session    : {sid}"]
    if m:
        L.append(f"armed      : YES  (since {m.get('armed_at')})")
        L.append(f"mode       : {m.get('mode')}   operator: {op or '(unknown)'}")
        L.append(f"debounce   : {m.get('debounce_min')}min   min_turns: {m.get('min_turns')}")
        tp = tp or m.get("transcript_path")
    else:
        L.append("armed      : no")
    let = e.get("last_extract_ts") or 0
    L.append(f"last summary: {iso(let) if let else '(never)'}")
    last_uuid = e.get("last_uuid") or "-"
    L.append(f"checkpoint : last_uuid={str(last_uuid)[:12]}  mem_v={e.get('last_mem_version', -1)}")
    if tp and Path(tp).exists():
        end, n = unprocessed_turns(tp, e.get("last_uuid"))
        up = "  (up to date)" if end == e.get("last_uuid") else ""
        L.append(f"unprocessed: {n} new human turn(s){up}")
    where = f"  in knowledge/{op}/" if op else "  (total)"
    L.append(f"records    : {record_count(op)}{where}")
    L.append(f"worker     : {'RUNNING (lock held)' if lock_path(sid).exists() else 'idle'}")
    return "\n".join(L)


def cmd_status(args):
    if args.all:
        ARMED_DIR.mkdir(parents=True, exist_ok=True)
        markers = sorted(ARMED_DIR.glob("*.json"))
        if not markers:
            print("(no armed sessions)")
            return 0
        for mp in markers:
            print(_status_one(mp.stem))
            print("-" * 52)
        return 0
    sid, tp = resolve_session(args.cwd, args.projects_base)
    if not sid:
        sys.stderr.write(f"no session for cwd={args.cwd}\n")
        return 1
    print(_status_one(sid, str(tp), args.cwd))
    print(f"hook       : {'installed' if hook_installed(args.cwd) else 'NOT installed'}")
    return 0


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def cmd_prep_done(args):
    sid, tp = resolve_session(args.cwd, args.projects_base)
    if not sid:
        sys.stderr.write(f"ERROR: cannot resolve session for cwd={args.cwd}\n")
        return 2
    m = read_marker(sid)
    cwd = (m or {}).get("cwd") or os.path.abspath(args.cwd)
    if m:
        mode, op = m.get("mode"), m.get("operator")
    else:
        mode, op = detect_mode_op(cwd, None)

    # Disarm first (monitoring off) — read marker above before removing it.
    mp = marker_path(sid)
    if mp.exists():
        mp.unlink()

    worker_running = lock_path(sid).exists()  # soft note; finalize holds global lock
    e = checkpoint_entry(sid)
    since = e.get("last_uuid") or ""
    freeze = last_uuid_of(Path(tp)) or ""
    prior_mv = int(e.get("last_mem_version", -1))
    mem_version = max(newest_mem_version(cwd), prior_mv)

    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    pending = PENDING_DIR / f"{sid}_final.md"

    cmd = [sys.executable, str(SCRIPTS / "extract_transcript.py"), str(tp)]
    if since:
        cmd += ["--since", since]
    if freeze:
        cmd += ["--freeze-at", freeze]
    cmd += ["--out", str(pending)]
    r = _run(cmd)
    if r.returncode != 0:
        sys.stderr.write(r.stderr or "extract_transcript failed\n")
        return 1

    if find_kernel_opt(cwd):
        dg = _run([sys.executable, str(SCRIPTS / "collect_workspace.py"),
                   "--cwd", cwd, "--since-version", str(prior_mv)])
        if dg.returncode == 0 and dg.stdout.strip():
            with open(pending, "a", encoding="utf-8") as fh:
                fh.write("\n\n---\n" + dg.stdout)

    _, new_turns = unprocessed_turns(tp, since)
    header = {
        "pending_path": str(pending),
        "session_id": sid,
        "operator": op,
        "mode": mode,
        "freeze_uuid": freeze,
        "mem_version": mem_version,
        "cwd": cwd,
        "since": since,
        "new_turns": new_turns,
        "worker_running": worker_running,
        "skill_dir": str(SKILL_DIR),
        "guides": list_templates("extraction"),      # read ALL for how-to
        "summaries": list_templates("summary"),       # pick the best-fit skeleton
    }
    print(json.dumps(header, ensure_ascii=False))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--cwd", default=os.getcwd())
        p.add_argument("--projects-base", type=Path, default=DEFAULT_PROJECTS_BASE)

    p = sub.add_parser("resolve"); common(p); p.set_defaults(fn=cmd_resolve)

    p = sub.add_parser("arm"); common(p)
    p.add_argument("--op", default="")
    p.add_argument("--debounce-min", type=float, default=DEFAULT_DEBOUNCE_MIN)
    p.add_argument("--min-turns", type=int, default=DEFAULT_MIN_TURNS)
    p.set_defaults(fn=cmd_arm)

    p = sub.add_parser("disarm"); common(p); p.set_defaults(fn=cmd_disarm)

    p = sub.add_parser("status"); common(p)
    p.add_argument("--all", action="store_true")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("prep-done"); common(p); p.set_defaults(fn=cmd_prep_done)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
