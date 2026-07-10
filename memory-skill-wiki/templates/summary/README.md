# templates/summary/ — Session-summary skeletons (auto-scanned)

**English** · [中文](README.zh.md)

Skeletons `/done-run` uses to write the **session summary**. `/done-run`
automatically reads **every `.md` in this folder** as an available skeleton; if
there are several, it picks the best fit for the operator / scenario and fills it in.

## Extend
- **Just drop a new skeleton `.md` here** (e.g. a different one per operator /
  hardware) — the next `/done-run` can choose it automatically: no code change, no
  reinstall.
- Editing an existing skeleton's content takes effect immediately.
- Remember to **re-upload to OSS** to sync.

## Scan rule
- Only `*.md` is read.
- **Ignored**: any `README*` file and files starting with `_` or `.`.
