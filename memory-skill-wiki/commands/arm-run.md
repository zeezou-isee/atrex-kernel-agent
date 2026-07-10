---
description: Arm kernel-experience monitoring for THIS session (auto-summarize at turn boundaries on new results / commits / kernel versions).
argument-hint: [operator_name]
allowed-tools: Bash(python3 __SKILL_DIR__/scripts/session_ctl.py:*)
---
Arm background kernel-optimization experience monitoring for the current session.
The operator is taken from your argument if given, else auto-detected from a
`kernel_opt_*` folder under the cwd, else inferred later by the summarizer.

Arming result:

!`python3 __SKILL_DIR__/scripts/session_ctl.py arm --cwd "$(pwd)" --op "$ARGUMENTS"`

Now tell the user, concisely:
- whether arming succeeded, and the detected **operator** + **mode** (`aka`/`vibe`);
- that from here on, at each turn boundary that is ≥ the debounce window apart AND
  shows new content (a new `memory/vN.json`, a perf/PASS-FAIL result line, a kernel
  `git commit`, or enough new turns), a silent background summary will be written
  under `knowledge/<operator>/`;
- if the output contains a **"Stop hook not found"** warning, surface the exact
  install command it printed and note the session must be **restarted** afterward
  for the hook to load.

Do not take any other action.
