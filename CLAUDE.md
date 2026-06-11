# hermes-agent (fork) — gateway/CLI platform ONLY

DGM-H code does NOT live here. The `dgmh/` package was removed on 2026-06-12
(see `dgmh/README.md` tombstone); the single source of truth is:

    ~/workspace/flask/python/dgmh_runtime/   (flask repo, ADR-004 + ADR-007)

Do NOT re-create or port dgmh modules into this repo — that duplication
previously caused silent live drift. This repo provides `hermes_cli` (the
gateway the live bot runs on) and its `venv/`. The gateway imports flask's
code via the `PYTHONPATH` drop-in on `hermes-gateway.service`.
