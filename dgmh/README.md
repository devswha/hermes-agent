# dgmh — MOVED (tombstone)

This package was consolidated into the flask repo as the single source of
truth on 2026-06-12 (flask ADR-004 + ADR-007):

    /home/devswha/workspace/flask/python/dgmh_runtime/

Do NOT re-create dgmh code here. The live gateway hooks
(`~/.hermes/hooks/*/handler.py`) import `dgmh_runtime.*` via the
`PYTHONPATH=.../flask/python` systemd drop-in; this repo only provides the
Hermes gateway/CLI platform (`hermes_cli`) and its venv.

Historical files (tests, fixtures, scripts, patina_profiles, skills,
HERMES_PATCH_NOTES.md, HERMES_REF.md) are recoverable from git history:

    git show 3145e35aa:dgmh/<path>      # last pre-consolidation state
