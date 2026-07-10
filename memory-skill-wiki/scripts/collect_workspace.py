#!/usr/bin/env python3
"""
collect_workspace.py — digest an atrex kernel_opt_* workspace's authoritative data.

The transcript rarely contains reliable performance numbers (large tool outputs
are offloaded to files, and do_bench/ncu results live in the tool layer). atrex,
however, writes the ground truth into kernel_opt_*/memory/v<N>.json (latency_us,
tflops, utilization %, rel_err, action_category, pitfalls...) and into README.md
(sourced Hardware Spec + Stop Conditions). This tool emits a compact Markdown
digest of that ground truth so Step 2 can attach real numbers to each experience.

Un-masked memory files only (masked==true means discarded — see gpu-kernel router).

Usage:
    collect_workspace.py --workspace DIR [--since-version N] [--out FILE]
    collect_workspace.py --cwd DIR    # auto-find kernel_opt_* under cwd
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def find_workspaces(root: Path):
    if not root or not root.is_dir():
        return []
    # a workspace is a dir that has memory/v*.json
    found = []
    for cand in sorted(root.glob("kernel_opt_*")):
        if (cand / "memory").is_dir():
            found.append(cand)
    # also allow root itself being the workspace
    if (root / "memory").is_dir():
        found.append(root)
    return found


def load_memories(ws: Path, since_version: int):
    mems = []
    for p in sorted((ws / "memory").glob("v*.json")):
        m = re.search(r"v(\d+)\.json$", p.name)
        if not m:
            continue
        ver = int(m.group(1))
        if ver <= since_version:
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if obj.get("masked") is True:
            continue
        obj["_ver"] = ver
        mems.append(obj)
    mems.sort(key=lambda o: o["_ver"])
    return mems


def readme_sections(ws: Path, headings):
    """Pull specific ## sections out of README.md (Hardware Spec / Stop Conditions)."""
    rp = ws / "README.md"
    if not rp.exists():
        return {}
    text = rp.read_text(encoding="utf-8", errors="replace")
    out = {}
    for h in headings:
        # match "## <h> ... up to next ## or EOF", case-insensitive on heading
        m = re.search(rf"(?im)^#{{1,4}}\s*{re.escape(h)}.*?$(.*?)(?=^#{{1,4}}\s|\Z)",
                      text, re.S | re.M)
        if m:
            body = m.group(1).strip()
            if body:
                out[h] = body[:1500]
    return out


def fmt_mem(m):
    perf = m.get("performance", {}) or {}
    corr = m.get("correctness", {}) or {}
    opt = m.get("optimization", {}) or {}
    prof = m.get("profile_evidence", {}) or {}
    cmp = perf.get("comparison_with_previous", {}) or {}
    bits = [f"**{m.get('version','v?')}**"]
    kv = []
    if perf.get("latency_us") is not None:
        kv.append(f"latency={perf['latency_us']}us")
    if perf.get("tflops") is not None:
        kv.append(f"tflops={perf['tflops']}")
    if perf.get("tflops_peak_utilization_pct") is not None:
        kv.append(f"tc_util={perf['tflops_peak_utilization_pct']}%")
    if perf.get("bandwidth_gbps") is not None:
        kv.append(f"bw={perf['bandwidth_gbps']}GB/s")
    if perf.get("bandwidth_peak_utilization_pct") is not None:
        kv.append(f"bw_util={perf['bandwidth_peak_utilization_pct']}%")
    if cmp.get("latency_delta") is not None:
        kv.append(f"Δlat={cmp['latency_delta']}")
    if corr.get("rel_err") is not None:
        kv.append(f"rel_err={corr['rel_err']}")
    if corr.get("status"):
        kv.append(f"status={corr['status']}")
    line = " · ".join(kv)
    parts = [f"- {' '.join(bits)}: {line}" if line else f"- {' '.join(bits)}"]
    if opt.get("action_category"):
        parts.append(f"    - action: `{opt['action_category']}` — {opt.get('action_description','') or ''}".rstrip(" —"))
    if opt.get("expected_impact"):
        parts.append(f"    - expected: {opt['expected_impact']}")
    if prof.get("bottleneck_type") or prof.get("evidence_summary"):
        parts.append(f"    - profile: {prof.get('bottleneck_type','')} {prof.get('evidence_summary','')}".rstrip())
    for pit in (m.get("pitfalls_and_fixes") or [])[:3]:
        if isinstance(pit, dict):
            les = pit.get("lesson") or pit.get("root_cause") or ""
            tag = pit.get("error_type", "")
        else:
            les, tag = str(pit), ""
        if les:
            parts.append(f"    - pitfall[{tag}]: {les}")
    return "\n".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", type=Path, default=None, help="explicit kernel_opt_* dir")
    ap.add_argument("--cwd", type=Path, default=None, help="dir to auto-scan for kernel_opt_*")
    ap.add_argument("--since-version", type=int, default=-1,
                    help="only include memory versions > this (incremental)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    workspaces = []
    if args.workspace:
        workspaces = [args.workspace]
    elif args.cwd:
        workspaces = find_workspaces(args.cwd)
    if not workspaces:
        msg = "_(no atrex workspace found; authoritative numbers unavailable)_\n"
        (args.out.write_text(msg, encoding="utf-8") if args.out else print(msg, end=""))
        return 0

    out = ["# atrex workspace ground-truth digest", ""]
    for ws in workspaces:
        mems = load_memories(ws, args.since_version)
        out.append(f"## workspace: `{ws.name}`  ({len(mems)} new iteration(s))")
        secs = readme_sections(ws, ["Hardware Spec", "Stop Conditions", "Task Context"])
        for h, body in secs.items():
            out.append(f"\n### {h}\n{body}")
        if mems:
            out.append("\n### Iteration ladder (authoritative numbers)")
            for m in mems:
                out.append(fmt_mem(m))
        out.append("")

    text = "\n".join(out).rstrip() + "\n"
    (args.out.write_text(text, encoding="utf-8") if args.out else print(text, end=""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
