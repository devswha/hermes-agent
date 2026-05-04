# DGM-H Hermes Patch Notes

Minimal upstream-eligible patches applied to Hermes core during W6 integration.
Re-apply these after rebasing onto upstream Hermes.

---

## Patch 1: Native-review gate (`run_agent.py::_spawn_background_review`)

**Rationale (plan v2 §CONCERN 3):**
While DGM-H is running and managing SOUL.md / skill evolution, the native
background review (`_spawn_background_review`) would write to the same
memory/skill stores, creating dueling writers and corrupting evolution state.

**Gate behaviour:**
- If `dgmh.gate_native_review: true` is present in `~/.hermes/config.yaml`,
  `_spawn_background_review` returns immediately without spawning a thread.
- The gate check is best-effort (try/except); if config.yaml is unreadable,
  the review proceeds normally.

**Patch hunk** (apply at the top of `_spawn_background_review`, after the docstring):

```python
# DGM-H native-review gate — upstream-eligible patch
try:
    import os as _os
    from pathlib import Path as _Path
    _hermes_home = _Path(_os.environ.get("HERMES_HOME") or (_Path.home() / ".hermes"))
    _cfg_path = _hermes_home / "config.yaml"
    if _cfg_path.exists():
        import yaml as _yaml
        _cfg = _yaml.safe_load(_cfg_path.read_text(encoding="utf-8")) or {}
        if (_cfg.get("dgmh") or {}).get("gate_native_review"):
            logger.debug(
                "DGM-H gate: skipping background review (dgmh.gate_native_review=true)"
            )
            return
except Exception:
    pass  # Gate check is best-effort; fall through if config unreadable
```

**Config flag** (added idempotently by `dgmh/hermes_integration/install_hooks.py`):

```yaml
dgmh:
  gate_native_review: true
```

**To disable the gate** (restore native review while keeping DGM-H installed):

```yaml
dgmh:
  gate_native_review: false
```

---

## No other core files patched.

The reaction hook is injected via the standard `~/.hermes/hooks/dgmh-reaction/`
plugin mechanism, not by modifying gateway core files.
