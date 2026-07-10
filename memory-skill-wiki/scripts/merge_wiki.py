#!/usr/bin/env python3
"""
merge_wiki.py (gpu-wiki-structured variant) — render knowledge/*.json into a wiki
whose layout and document shapes MATCH atrex-kernel-agent/gpu-wiki, so the output
can be dropped straight into that knowledge base.

Layout produced under wiki/ (mirrors gpu-wiki's three-level vendor→dsl→arch tree):

  wiki/README.md                                                  top routing index
  wiki/docs/kernel-opt/<vendor>/<dsl>/<arch>/<topic>.md           positive: Trigger + technique set
  wiki/docs/kernel-opt/<vendor>/<dsl>/<arch>/README.md            index table
  wiki/docs/pitfalls/<vendor>/<dsl>/<arch>/<topic>-pitfalls.md    negative: Trap/Result/Why/Lesson (5-step)
  wiki/docs/pitfalls/<vendor>/<dsl>/<arch>/README.md              index table (File | Kernel | Hardware | Trap count)
  wiki/docs/ref-docs/<vendor>/<dsl>/<arch>/<topic>-optimization.md  full version-by-version journey

Deterministic: same records -> same bytes (dates come from records' extracted_at,
not the clock). The per-record five-part `body` maps onto gpu-wiki sections:
  方法→Trap/Technique · 实测→Result/Effect · 归因与结论→Why/Lesson.

Usage: merge_wiki.py [--knowledge DIR] [--wiki DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

OUTCOME_ORDER = {"positive": 0, "neutral": 1, "negative": 2}
OUTCOME_BADGE = {"positive": "✅", "neutral": "➖", "negative": "❌"}


def slugify(s: str) -> str:
    s = (s or "unknown").strip().lower()
    s = re.sub(r"[^\w.-]+", "-", s)
    return s.strip("-") or "unknown"


def load_records(kdir: Path):
    recs = []
    # rglob so per-operator subfolders (knowledge/<op>/*.json) are picked up; the
    # flat legacy layout still works. Skip the tracking git repo and .gitkeep.
    for p in sorted(kdir.rglob("*.json")):
        if ".git" in p.parts:
            continue
        try:
            o = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"  ! skip {p.name}: {e}")
            continue
        o["_file"] = p.name
        recs.append(o)
    return recs


# ---- routing: derive vendor / dsl / arch / topic from a record ---------------

def hw(rec):
    h = (rec.get("tags", {}) or {}).get("hardware", {})
    return h if isinstance(h, dict) else {}


def platform_of(rec):
    p = hw(rec).get("platform") or []
    if p:
        return p[0]
    return ((rec.get("context", {}) or {}).get("hardware", "") or "").split("/")[0].strip()


def vendor_of(rec):
    p = platform_of(rec).upper()
    a = " ".join(hw(rec).get("arch") or []).upper()
    t = " ".join(hw(rec).get("target") or []).upper()
    if p.startswith("MI") or "CDNA" in a or "GFX" in t:
        return "amd"
    return "nvidia"


def dsl_slug(rec):
    d = ((rec.get("context", {}) or {}).get("dsl", "") or "").lower()
    if "cute" in d:
        return "cutedsl"
    if "fly" in d:
        return "flydsl"
    if "triton" in d:
        return "triton"
    if "gluon" in d:
        return "gluon"
    return "common"


# platform -> compute-capability directory (unambiguous; each chip has one CC).
# B300 = sm103 (Blackwell Ultra, sm_103/sm_100a), B200/GB200 = sm100, etc.
PLATFORM_ARCH = {
    "B300": "sm103", "GB300": "sm103",
    "B200": "sm100", "GB200": "sm100",
    "H100": "sm90", "H200": "sm90", "H800": "sm90", "H20": "sm90",
    "A100": "sm80",
    "MI300X": "gfx942", "MI308X": "gfx942", "MI325X": "gfx942",
    "MI355X": "gfx950", "MI250X": "gfx90a",
}


def arch_slug(rec):
    # 1) platform is unambiguous -> prefer it (B300 -> sm103, not sm100)
    plat = platform_of(rec).upper()
    if plat in PLATFORM_ARCH:
        return PLATFORM_ARCH[plat]
    # 2) explicit compute-capability target tag, most-specific first
    tgt = " ".join(hw(rec).get("target") or []).lower()
    for k in ("sm103", "sm120", "sm100a", "sm100", "sm90a", "sm90",
              "sm89", "sm86", "sm80", "gfx950", "gfx942", "gfx90a"):
        if k in tgt:
            return {"sm100a": "sm100", "sm90a": "sm90"}.get(k, k)  # normalize -a suffix
    # 3) arch family fallback
    arch = " ".join(hw(rec).get("arch") or []).lower()
    first = arch.split()[0] if arch else ""
    return {"blackwell": "sm100", "hopper": "sm90", "ampere": "sm80",
            "cdna3": "gfx942", "cdna4": "gfx950"}.get(first, "common")


def topic_of(rec):
    return slugify((rec.get("context", {}) or {}).get("operator", "unknown"))


def route(rec):
    return (vendor_of(rec), dsl_slug(rec), arch_slug(rec), topic_of(rec))


def version_of(rec):
    m = re.search(r"[_/]v(\d+)\b", rec.get("id", ""))
    return int(m.group(1)) if m else None


def hw_label(rec):
    h = hw(rec)
    bits = (h.get("platform") or []) + (h.get("arch") or []) + (h.get("target") or [])
    return " / ".join(dict.fromkeys(bits)) or "?"


def techniques(rec):
    cat = (rec.get("tags", {}) or {}).get("category", {})
    if isinstance(cat, dict):
        return list(cat.get("technique", []) or [])
    return list(cat or [])


def bottlenecks(rec):
    return list((rec.get("tags", {}) or {}).get("bottleneck", []) or [])


def outcome_of(rec):
    return (rec.get("attempt", {}) or {}).get("outcome", "neutral")


def body_parts(rec):
    """Split the five-part body into labeled parts by its Chinese labels."""
    body = rec.get("body", "") or ""
    labels = [("setting", "工况"), ("method", "方法"), ("expect", "预期"),
              ("result", "实测"), ("attrib", "归因")]
    idx = []
    for key, lab in labels:
        m = re.search(rf"{lab}[^:：]*[:：]", body)
        if m:
            idx.append((m.start(), m.end(), key))
    idx.sort()
    out = {}
    for i, (s, e, key) in enumerate(idx):
        end = idx[i + 1][0] if i + 1 < len(idx) else len(body)
        out[key] = body[e:end].strip()
    return out


def title_of(rec):
    """Clean header from the id: drop the leading <operator>_<platform> prefix."""
    parts = [p for p in re.split(r"[_/]", rec.get("id", "")) if p]
    if len(parts) > 2:
        parts = parts[2:]  # drop operator-abbrev + platform tokens
    title = " ".join(parts).strip()
    return title or rec.get("id", "") or "(untitled)"


def src_line(rec):
    s = rec.get("source", {}) or {}
    bits = []
    if s.get("session_id"):
        bits.append(f"session `{str(s['session_id'])[:8]}`")
    if s.get("git_commit"):
        bits.append(f"commit `{s['git_commit']}`")
    tg = techniques(rec) + bottlenecks(rec)
    tagstr = ("`" + "` `".join(tg) + "`") if tg else ""
    return "  ".join(x for x in [tagstr, " · ".join(bits)] if x)


def latest_date(recs):
    ds = [(r.get("source", {}) or {}).get("extracted_at", "") for r in recs]
    ds = [d for d in ds if d]
    return max(ds)[:10] if ds else "n/a"


def kernel_desc(recs):
    c = (recs[0].get("context", {}) or {})
    return f"`{c.get('operator','?')}` — {c.get('dsl','?')}, {c.get('dtype','?')}, {c.get('shapes','?')}"


def rel(from_path: Path, to_path: Path):
    return os.path.relpath(to_path, from_path.parent)


# ---- document renderers ------------------------------------------------------

def render_pitfall_entry(n, rec):
    a = rec.get("attempt", {}) or {}
    bp = body_parts(rec)
    trap = a.get("method", "") or bp.get("method", "")
    exp = a.get("expected", "") or bp.get("expect", "")
    result = a.get("measured", "") or bp.get("result", "")
    why = a.get("reason", "")
    lesson = bp.get("attrib", "") or why
    L = [f"## {n}. {title_of(rec)}", ""]
    L.append(f"**Trap**: {trap}" + (f"（预期 {exp}）" if exp else ""))
    L.append("")
    L.append(f"**Result**: {result}")
    L.append("")
    if why:
        L.append(f"**Why**: {why}")
        L.append("")
    L.append(f"**Lesson**: {lesson}")
    L.append("")
    sl = src_line(rec)
    if sl:
        L.append(f"<sub>{sl}</sub>")
        L.append("")
    return "\n".join(L)


def render_kernelopt_entry(rec):
    a = rec.get("attempt", {}) or {}
    c = rec.get("context", {}) or {}
    bn = bottlenecks(rec)
    L = [f"## {title_of(rec)}", ""]
    trig = ", ".join(bn) if bn else "—"
    wl = " / ".join(x for x in [c.get("workload", ""), c.get("shapes", "")] if x)
    L.append(f"**Trigger**: {trig}" + (f" — {wl}" if wl else ""))
    L.append("")
    L.append(f"**Technique**: {a.get('method','')}")
    L.append("")
    if a.get("expected"):
        L.append(f"**Expected**: {a['expected']}")
        L.append("")
    L.append(f"**Effect**: {a.get('measured','')}")
    L.append("")
    if a.get("reason"):
        L.append(f"**Why it works**: {a['reason']}")
        L.append("")
    sl = src_line(rec)
    if sl:
        L.append(f"<sub>{sl}</sub>")
        L.append("")
    return "\n".join(L)


def render_journey(recs, topic, arch):
    a0 = recs[0]
    c = a0.get("context", {}) or {}
    L = [f"# {topic.replace('_',' ').title()} on {arch} — Optimization Journey (ref-docs)", ""]
    L.append(f"**Last Updated**: {latest_date(recs)}  ·  {len(recs)} recorded attempt(s)")
    L.append("")
    L.append(f"Hardware: {hw_label(a0)} · DSL: {c.get('dsl','?')} · dtype: {c.get('dtype','?')} · "
             f"shapes: {c.get('shapes','?')}")
    L.append("")
    L.append("---")
    L.append("")
    versioned = sorted([r for r in recs if version_of(r) is not None],
                       key=lambda r: (version_of(r), OUTCOME_ORDER.get(outcome_of(r), 1)))
    if versioned:
        L.append("## Version ladder")
        L.append("")
        L.append("| Ver | Outcome | Technique | Measured |")
        L.append("|-----|---------|-----------|----------|")
        for r in versioned:
            a = r.get("attempt", {}) or {}
            badge = OUTCOME_BADGE.get(outcome_of(r), "")
            tech = ", ".join(techniques(r)[:3]) or "—"
            meas = (a.get("measured", "") or "").replace("\n", " ")[:90]
            L.append(f"| v{version_of(r)} | {badge} | {tech} | {meas} |")
        L.append("")
    L.append("## Attempts in detail")
    L.append("")
    ordered = sorted(recs, key=lambda r: (version_of(r) if version_of(r) is not None else 999,
                                          OUTCOME_ORDER.get(outcome_of(r), 1)))
    for r in ordered:
        a = r.get("attempt", {}) or {}
        badge = OUTCOME_BADGE.get(outcome_of(r), "")
        vtag = f"v{version_of(r)} · " if version_of(r) is not None else ""
        L.append(f"### {badge} {vtag}{title_of(r)}")
        L.append("")
        body = (r.get("body") or "").strip()
        if body:
            L.append(body)
        else:
            for lab, key in (("方法", "method"), ("预期", "expected"),
                             ("实测", "measured"), ("归因", "reason")):
                if a.get(key):
                    L.append(f"- **{lab}**: {a[key]}")
        L.append("")
        sl = src_line(r)
        if sl:
            L.append(f"<sub>{sl}</sub>")
        L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--knowledge", type=Path, default=ROOT / "knowledge")
    ap.add_argument("--wiki", type=Path, default=ROOT / "wiki")
    args = ap.parse_args(argv)

    recs = load_records(args.knowledge)
    # fully wipe the generated tree (files AND empty dirs) so no stale arch dirs linger
    docs_root = args.wiki / "docs"
    if docs_root.exists():
        shutil.rmtree(docs_root)
    top_readme = args.wiki / "README.md"
    if top_readme.exists():
        top_readme.unlink()

    groups = defaultdict(list)
    for r in recs:
        groups[route(r)].append(r)

    pitfall_docs = defaultdict(list)   # (vendor,dsl,arch) -> [(path, items, negs)]
    kopt_docs = defaultdict(list)
    index_rows = []

    for (vendor, dsl, arch, topic), items in sorted(groups.items()):
        items.sort(key=lambda r: (version_of(r) if version_of(r) is not None else 999,
                                  OUTCOME_ORDER.get(outcome_of(r), 1)))
        negs = [r for r in items if outcome_of(r) == "negative"]
        poss = [r for r in items if outcome_of(r) == "positive"]

        base = f"{vendor}/{dsl}/{arch}"
        ko_dir = args.wiki / "docs" / "kernel-opt" / base
        pf_dir = args.wiki / "docs" / "pitfalls" / base
        rd_dir = args.wiki / "docs" / "ref-docs" / base
        for d in (rd_dir,):
            d.mkdir(parents=True, exist_ok=True)

        journey = rd_dir / f"{topic}-optimization.md"
        journey.write_text(render_journey(items, topic, arch) + "\n", encoding="utf-8")

        ko = ko_dir / f"{topic}.md"
        pf = pf_dir / f"{topic}-pitfalls.md"

        if poss:
            ko_dir.mkdir(parents=True, exist_ok=True)
            head = [f"# {arch} {topic.replace('_',' ')} — Optimization Highlights (kernel-opt)", "",
                    "> Optimization highlights (one entry per proven technique). "
                    "Full version-by-version journey + pitfalls: see Further Reading.", "",
                    f"**Further Reading**: [journey]({rel(ko, journey)})"
                    + (f" · [pitfalls]({rel(ko, pf)})" if negs else ""),
                    "", "---", ""]
            body = "\n".join(render_kernelopt_entry(r) for r in poss)
            ko.write_text("\n".join(head) + "\n" + body + "\n", encoding="utf-8")
            kopt_docs[(vendor, dsl, arch)].append((ko, items, poss))

        if negs:
            pf_dir.mkdir(parents=True, exist_ok=True)
            head = [f"# {topic.replace('_',' ').title()} on {arch} — Pitfalls", "",
                    f"Traps hit while optimizing {kernel_desc(items)} on {hw_label(items[0])}.",
                    "Companion to:", "",
                    f"- Optimization journey: [journey]({rel(pf, journey)})"]
            if poss:
                head.append(f"- Optimization highlights: [highlights]({rel(pf, ko)})")
            head += ["", "---", ""]
            body = "\n".join(render_pitfall_entry(i + 1, r) for i, r in enumerate(negs))
            pf.write_text("\n".join(head) + "\n" + body + "\n", encoding="utf-8")
            pitfall_docs[(vendor, dsl, arch)].append((pf, items, negs))

        index_rows.append((vendor, dsl, arch, topic, len(items), len(poss), len(negs),
                           journey, ko if poss else None, pf if negs else None))

    # per-leaf README index tables (gpu-wiki style)
    for (vendor, dsl, arch), docs in pitfall_docs.items():
        d = args.wiki / "docs" / "pitfalls" / f"{vendor}/{dsl}/{arch}"
        L = [f"# {arch} {dsl} Pitfalls ({vendor})", "",
             "| File | Kernel | Hardware | Trap count |", "|------|--------|----------|-----------|"]
        for path, items, negs in sorted(docs, key=lambda x: x[0].name):
            L.append(f"| [{path.name}]({path.name}) | {kernel_desc(items)} | {hw_label(items[0])} | {len(negs)} |")
        (d / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    for (vendor, dsl, arch), docs in kopt_docs.items():
        d = args.wiki / "docs" / "kernel-opt" / f"{vendor}/{dsl}/{arch}"
        L = [f"# {arch} {dsl} Optimization Highlights ({vendor})", "",
             "| File | Kernel | Hardware | Techniques |", "|------|--------|----------|-----------|"]
        for path, items, poss in sorted(docs, key=lambda x: x[0].name):
            L.append(f"| [{path.name}]({path.name}) | {kernel_desc(items)} | {hw_label(items[0])} | {len(poss)} |")
        (d / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # top routing README
    top = ["# Kernel-Experience Wiki (gpu-wiki structured)", "",
           "Distilled GPU-kernel-optimization experience, laid out to match "
           "`atrex-kernel-agent/gpu-wiki` (vendor → DSL → architecture). Each topic has up to "
           "three docs: **kernel-opt** (proven techniques), **pitfalls** (Trap→Result→Why→Lesson), "
           "and **ref-docs** (full version journey).", "",
           f"_{len(recs)} experience record(s) across {len(groups)} topic group(s)._", "",
           "| Vendor | DSL | Arch | Topic | Records | ✅ | ❌ | kernel-opt | pitfalls | journey |",
           "|--------|-----|------|-------|--------:|---:|---:|-----------|----------|---------|"]
    readme = args.wiki / "README.md"
    for (vendor, dsl, arch, topic, n, npos, nneg, journey, kop, pfp) in sorted(index_rows):
        kolink = f"[✓]({rel(readme, kop)})" if kop else "—"
        pflink = f"[✓]({rel(readme, pfp)})" if pfp else "—"
        jlink = f"[✓]({rel(readme, journey)})"
        top.append(f"| {vendor} | {dsl} | {arch} | {topic} | {n} | {npos} | {nneg} | {kolink} | {pflink} | {jlink} |")
    readme.write_text("\n".join(top) + "\n", encoding="utf-8")

    ndocs = len(list(args.wiki.rglob("*.md")))
    print(f"wrote {ndocs} markdown file(s) under {args.wiki} (gpu-wiki structure) from {len(recs)} record(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
