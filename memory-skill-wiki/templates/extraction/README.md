# templates/extraction/ — Extraction guides (auto-scanned)

**English** · [中文](README.zh.md)

Guides the extractor follows **when writing records**. Both extraction paths — the
background worker (`run_extraction.sh`) and the final `/done-run` — automatically
read **every `.md` in this folder** as the "how to write" spec.

## Extend
- **Just drop a new `.md` here** — the next extraction picks it up automatically:
  no code change, no reinstall.
- Editing an existing guide's content takes effect immediately (the skill dir is
  symlinked, so scripts read the current content at runtime).
- Remember to **re-upload to OSS** to sync (local edits don't sync automatically).

## Scan rule
- Only `*.md` is read.
- **Ignored**: any `README*` file and files starting with `_` or `.`
  (to temporarily disable a guide, prefix its name with `_`).
