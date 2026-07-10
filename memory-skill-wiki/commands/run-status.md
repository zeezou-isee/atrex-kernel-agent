---
description: Show kernel-experience monitoring status for THIS session (armed? operator, checkpoint, unprocessed slice, records, worker, hook).
allowed-tools: Bash(python3 __SKILL_DIR__/scripts/session_ctl.py:*)
---
Report the kernel-experience monitoring status for the current session.

!`python3 __SKILL_DIR__/scripts/session_ctl.py status --cwd "$(pwd)"`

Relay the above to the user as-is (it is already formatted). If it shows
`armed: no`, remind them they can start monitoring with `/arm-run [operator]`. If
`hook: NOT installed`, note that monitoring cannot fire until the hook is
installed (`install.sh --global`) and the session is restarted. Do not take any
other action.
