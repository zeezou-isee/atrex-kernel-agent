#!/usr/bin/env python3
"""
extract_transcript.py — Step 1 of the kernel-experience extraction skill.

Reads a Claude Code session transcript (.jsonl), filters out tool noise, splits
the conversation into human turns, inlines the *work* sub-agent transcripts, and
emits a clean normalized Markdown document that Step 2 (the LLM) turns into
structured, tagged experience records.

Design notes (see plan):
  - Turn boundary = a real human message (promptSource in typed/queued/
    suggestion_accepted). Everything until the next human message belongs to it.
  - Kept content: human prompts, assistant `text`, assistant `thinking`.
    Dropped: `tool_use` inputs and `tool_result` outputs (~70% of the volume).
    A compact per-turn "tool tally" line is kept for context (not the payloads).
  - Sub-agents live in <session_dir>/<session_id>/subagents/agent-*.jsonl with a
    sibling agent-*.meta.json {agentType, description, toolUseId}. We inline the
    *work* sub-agents (baseline/optimizer/Explore/...) at the Task/Agent tool_use
    that spawned them, but EXCLUDE our own extraction sub-agents by agentType /
    description prefix (--exclude-agent, repeatable; sensible defaults below).
  - Incremental / concurrency: --since <uuid> (exclusive lower bound) and
    --freeze-at <uuid> (inclusive upper bound) slice the append-only main chain.

Usage:
    extract_transcript.py SESSION.jsonl [--since UUID] [--freeze-at UUID]
        [--subagents-dir DIR] [--exclude-agent SUBSTR ...]
        [--tool-tally/--no-tool-tally] [--out FILE]

A machine-readable trailer is written to stderr:
    META {"session_id":..., "first_uuid":..., "last_uuid":..., "n_turns":...}
so callers (checkpoint.py / the hook) can advance the checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

# Human-authored prompt sources = turn boundaries.
HUMAN_SOURCES = {"typed", "queued", "suggestion_accepted"}
# Sub-agent identities we always skip (our own extraction machinery).
DEFAULT_EXCLUDE = ["memory", "memory-skill", "kernel-experience", "experience-extract"]
# Tool names that spawn a sub-agent.
SPAWN_TOOLS = {"Task", "Agent"}

# Only salvage "产出的结果" from result-PRODUCING tools. Benchmarks / tests /
# profilers all run through Bash; Read/Grep/Glob/LS/Edit dumps are pure noise and
# must stay filtered. (Task sub-agents are inlined separately.)
RESULT_TOOLS = {"Bash"}

# A result-signal line must carry a NUMBER+unit, an explicit metric key, an
# explicit PASS/FAIL, a speedup, or a hard error. Bare prose words (latency,
# correct, registers, nan…) are intentionally excluded to avoid matching code.
RESULT_SIGNAL_RE = re.compile(
    r"(\b\d+(?:\.\d+)?\s*(?:µs|us|ms|ns|TFLOPS|GFLOPS|GB/?s|TB/?s|%)\b"
    r"|\b(?:rel_err|rel_l2|abs_err|max_?err|latency_us|tflops)\s*[=:]"
    r"|\b(?:PASS|FAIL|PASSED|FAILED)\b"
    r"|\b\d+(?:\.\d+)?\s*[x×](?=\s|$|[),])|speedup"
    r"|Traceback|Error:|\bException\b|CUDA error|illegal memory|OOM|out of memory)")


def load_jsonl(path: Path):
    """Yield parsed JSON objects from a .jsonl file, skipping blank/bad lines."""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def slice_by_uuid(records, since=None, freeze_at=None):
    """Return records in (since, freeze_at]; bounds matched on the `uuid` field.

    since is exclusive (already-processed), freeze_at is inclusive (frozen end).
    If a bound uuid is not found it is ignored (fail-open, so we never drop data
    silently on a stale checkpoint)."""
    out = list(records)
    if since:
        for i, r in enumerate(out):
            if r.get("uuid") == since:
                out = out[i + 1:]
                break
    if freeze_at:
        for i, r in enumerate(out):
            if r.get("uuid") == freeze_at:
                out = out[: i + 1]
                break
    return out


def blocks(msg):
    """Normalize a message's content into a list of blocks."""
    if not isinstance(msg, dict):
        return []
    content = msg.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return content
    return []


def human_text(record):
    """Return the human prompt text if this record is a real human turn, else None."""
    if record.get("type") != "user":
        return None
    if record.get("promptSource") not in HUMAN_SOURCES:
        return None
    msg = record.get("message", {})
    content = msg.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content if b.get("type") == "text"]
        joined = "\n".join(p for p in parts if p).strip()
        return joined or None
    return None


def index_subagents(subagents_dir: Path, exclude_substrs):
    """Map toolUseId -> {agentType, description, records[]} for *work* sub-agents.

    Sub-agents whose agentType or description contains any exclude substring
    (case-insensitive) are dropped — that is how we avoid re-ingesting our own
    extraction runs."""
    index = {}
    if not subagents_dir.is_dir():
        return index
    for meta_path in sorted(subagents_dir.glob("agent-*.meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        agent_type = str(meta.get("agentType", ""))
        desc = str(meta.get("description", ""))
        hay = (agent_type + " " + desc).lower()
        if any(sub.lower() in hay for sub in exclude_substrs):
            continue
        tool_use_id = meta.get("toolUseId")
        if not tool_use_id:
            continue
        jsonl_path = meta_path.with_suffix("").with_suffix(".jsonl")
        if not jsonl_path.exists():
            jsonl_path = meta_path.parent / (meta_path.name.replace(".meta.json", ".jsonl"))
        records = list(load_jsonl(jsonl_path)) if jsonl_path.exists() else []
        index[tool_use_id] = {
            "agentType": agent_type,
            "description": desc,
            "records": records,
        }
    return index


def result_lines(msg, id2name, per_block_cap=8):
    """Extract result-signal lines from a message's Bash tool_result blocks.

    Only tool_results whose originating tool (via tool_use_id -> id2name) is a
    RESULT_TOOL are scanned; only lines matching RESULT_SIGNAL_RE are kept; blobs
    and cat -n file lines are skipped; capped per block. Returns [] for pure-noise
    results so we honor '过滤无关信息' while preserving '产出的结果'."""
    kept = []
    for b in blocks(msg):
        if b.get("type") != "tool_result":
            continue
        if id2name.get(b.get("tool_use_id")) not in RESULT_TOOLS:
            continue
        c = b.get("content")
        if isinstance(c, list):
            text = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
        else:
            text = str(c or "")
        hits = []
        for ln in text.splitlines():
            ln = ln.strip()
            if not ln or len(ln) > 400 or re.match(r"^\d+\t", ln):
                continue  # skip blobs and cat -n file dumps
            if RESULT_SIGNAL_RE.search(ln):
                hits.append(ln)
            if len(hits) >= per_block_cap:
                hits.append("… (result truncated)")
                break
        kept.extend(hits)
    return kept


def render_assistant_blocks(record, out_lines, tool_counter, spawned_ids, id2name):
    """Append kept assistant content (text + thinking) to out_lines.

    Records tool usage into tool_counter, maps tool_use id -> name (for later
    result salvage), and collects sub-agent spawn tool ids."""
    for b in blocks(record.get("message", {})):
        btype = b.get("type")
        if btype == "text":
            txt = (b.get("text") or "").strip()
            if txt:
                out_lines.append(txt)
        elif btype == "thinking":
            th = (b.get("thinking") or "").strip()
            if th:
                out_lines.append("> [thinking] " + th.replace("\n", "\n> "))
        elif btype == "tool_use":
            name = b.get("name", "?")
            tool_counter[name] += 1
            if b.get("id"):
                id2name[b["id"]] = name
            if name in SPAWN_TOOLS and b.get("id"):
                spawned_ids.append(b.get("id"))


def render_subagent(sub, out_lines):
    """Inline a work sub-agent's filtered text/thinking under the current turn."""
    at = sub.get("agentType") or "subagent"
    desc = sub.get("description") or ""
    out_lines.append(f"\n<details:subagent {at}> {desc}".rstrip())
    tally = Counter()
    body = []
    id2name = {}
    seen_prompt = False
    for rec in sub.get("records", []):
        rtype = rec.get("type")
        if rtype == "assistant":
            spawned = []  # nested spawns not inlined (one level deep is enough)
            render_assistant_blocks(rec, body, tally, spawned, id2name)
        elif rtype == "user":
            msg = rec.get("message", {})
            c = msg.get("content")
            # The sub-agent's first user string message IS its task prompt
            # (promptSource is null inside sub-agents, so match on position).
            if not seen_prompt and isinstance(c, str) and c.strip():
                body.append(f"[task prompt] {c.strip()}")
                seen_prompt = True
                continue
            # Otherwise keep only result signals from its Bash tool_results.
            for rl in result_lines(msg, id2name):
                body.append(f"[result] {rl}")
    if tally:
        body.append("_tools: " + ", ".join(f"{k}×{v}" for k, v in tally.most_common()) + "_")
    out_lines.extend(body if body else ["_(no textual content)_"])
    out_lines.append("</details:subagent>")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path, help="path to main session .jsonl")
    ap.add_argument("--since", default=None, help="exclusive lower-bound uuid (already processed)")
    ap.add_argument("--freeze-at", default=None, help="inclusive upper-bound uuid (frozen end)")
    ap.add_argument("--subagents-dir", type=Path, default=None,
                    help="override sub-agents dir (default: <dir>/<stem>/subagents)")
    ap.add_argument("--exclude-agent", action="append", default=None,
                    help="skip sub-agents whose agentType/description contains this (repeatable)")
    ap.add_argument("--no-tool-tally", dest="tool_tally", action="store_false",
                    help="omit the per-turn compact tool tally line")
    ap.add_argument("--no-result-signals", dest="result_signals", action="store_false",
                    help="drop tool_result result-signal lines entirely (default: keep)")
    ap.add_argument("--out", type=Path, default=None, help="output file (default stdout)")
    ap.set_defaults(tool_tally=True, result_signals=True)
    args = ap.parse_args(argv)

    if not args.session.exists():
        ap.error(f"session not found: {args.session}")

    exclude = args.exclude_agent if args.exclude_agent is not None else list(DEFAULT_EXCLUDE)

    records = slice_by_uuid(load_jsonl(args.session), args.since, args.freeze_at)

    # Locate sub-agents dir: <session_dir>/<stem>/subagents by default.
    stem = args.session.name[:-6] if args.session.name.endswith(".jsonl") else args.session.stem
    sub_dir = args.subagents_dir or (args.session.parent / stem / "subagents")
    sub_index = index_subagents(sub_dir, exclude)

    session_id = next((r.get("sessionId") for r in records if r.get("sessionId")), stem)
    cwd = next((r.get("cwd") for r in records if r.get("cwd")), None)
    uuids = [r.get("uuid") for r in records if r.get("uuid")]
    first_uuid = uuids[0] if uuids else None
    last_uuid = uuids[-1] if uuids else None

    out = [f"# Session transcript (filtered)\n",
           f"- session_id: `{session_id}`",
           f"- source: `{args.session.name}`",
           f"- range: since=`{args.since or 'BEGIN'}` .. freeze=`{args.freeze_at or 'END'}`",
           f"- work sub-agents inlined: {len(sub_index)} (excluded substrings: {exclude})\n"]

    turn_no = 0
    turn_lines = []
    turn_tools = Counter()
    turn_spawns = []
    id2name = {}
    have_turn = False

    def flush_turn():
        nonlocal turn_lines, turn_tools, turn_spawns, have_turn
        if not have_turn:
            return
        if args.tool_tally and turn_tools:
            out.append("_tools: " + ", ".join(f"{k}×{v}" for k, v in turn_tools.most_common()) + "_\n")
        # inline work sub-agents spawned in this turn
        for tid in turn_spawns:
            sub = sub_index.get(tid)
            if sub:
                render_subagent(sub, out)
        out.extend(turn_lines)
        out.append("")
        turn_lines = []
        turn_tools = Counter()
        turn_spawns = []
        have_turn = False

    for rec in records:
        ht = human_text(rec)
        if ht is not None:
            flush_turn()
            turn_no += 1
            have_turn = True
            ts = rec.get("timestamp", "")
            out.append(f"\n## 回合 {turn_no}  ·  {ts}")
            out.append(f"**[human]** {ht}\n")
            continue
        if rec.get("type") == "assistant":
            if not have_turn:
                # assistant content before the first human prompt (e.g. resumed session)
                turn_no += 1
                have_turn = True
                out.append(f"\n## 回合 {turn_no} (无显式人类输入)")
            render_assistant_blocks(rec, turn_lines, turn_tools, turn_spawns, id2name)
        elif rec.get("type") == "user" and have_turn and args.result_signals:
            # keep only result signals (numbers / PASS-FAIL / errors) from Bash tool_results
            for rl in result_lines(rec.get("message", {}), id2name):
                turn_lines.append(f"`[result]` {rl}")
        # all other records (system, attachment, pure-noise tool_results) are dropped

    flush_turn()

    text = "\n".join(out).rstrip() + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)

    meta = {"session_id": session_id, "cwd": cwd, "first_uuid": first_uuid,
            "last_uuid": last_uuid, "n_turns": turn_no}
    sys.stderr.write("META " + json.dumps(meta, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
