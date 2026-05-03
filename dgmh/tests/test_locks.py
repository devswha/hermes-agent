"""
dgmh/tests/test_locks.py — Tests for locks.py per-skill flock.

Verifies:
- acquire_skill_lock context manager yields and releases cleanly
- lock file is created under ~/.hermes/dgmh/locks/
- concurrent-write test: 2 subprocesses competing for same lock; second
  observes consistent state (never tears the resource)
- timeout exhaustion raises SkillLockError and logs archive-lock-error
- SkillLockError carries category/name

Reference: hermes-skill-archive-dgmh-plan-v2-addendum.md §R-3 (lock scope)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dgmh.locks import SkillLockError, acquire_skill_lock, _lock_file_path


# ---------------------------------------------------------------------------
# Basic context manager
# ---------------------------------------------------------------------------

class TestAcquireSkillLock:
    def test_acquires_and_releases(self, tmp_path):
        """Lock acquired inside context, released on exit."""
        with patch_hermes_home(tmp_path):
            with acquire_skill_lock("cat-a", "skill-x") as lock_path:
                assert lock_path.exists()
            # After context exits, lock released (no error)

    def test_lock_file_path_format(self, tmp_path):
        """Lock file uses <cat>__<name>.lock naming."""
        with patch_hermes_home(tmp_path):
            path = _lock_file_path("my-cat", "my-skill")
        assert path.name == "my-cat__my-skill.lock"

    def test_lock_dir_created_automatically(self, tmp_path):
        """Lock directory is created if it doesn't exist."""
        with patch_hermes_home(tmp_path):
            lock_dir = tmp_path / "dgmh" / "locks"
            assert not lock_dir.exists()
            with acquire_skill_lock("cat-a", "skill-x"):
                assert lock_dir.exists()

    def test_sequential_locks_work(self, tmp_path):
        """Same skill can be locked sequentially (no deadlock)."""
        with patch_hermes_home(tmp_path):
            for _ in range(3):
                with acquire_skill_lock("cat-a", "skill-x"):
                    pass  # no error

    def test_different_skills_lock_independently(self, tmp_path):
        """Two different skills can hold locks simultaneously (no conflict)."""
        with patch_hermes_home(tmp_path):
            with acquire_skill_lock("cat-a", "skill-x"):
                # While holding cat-a/skill-x, also acquire cat-b/skill-y
                with acquire_skill_lock("cat-b", "skill-y"):
                    pass

    def test_timeout_exhaustion_raises_skill_lock_error(self, tmp_path):
        """Timeout exhaustion raises SkillLockError with correct category/name."""
        import fcntl

        with patch_hermes_home(tmp_path):
            # Compute lock path inside patched env
            lock_path = _lock_file_path("cat-a", "skill-conflict")
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path.touch()

            # Pre-acquire an exclusive lock on the file to force contention
            blocker_fd = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY, 0o600)
            fcntl.flock(blocker_fd, fcntl.LOCK_EX)
            try:
                with pytest.raises(SkillLockError) as exc_info:
                    # Very short timeout; retries=1 so it fails quickly
                    with acquire_skill_lock(
                        "cat-a",
                        "skill-conflict",
                        timeout_s=0.1,
                        retries=1,
                    ):
                        pass
                err = exc_info.value
                assert err.category == "cat-a"
                assert err.name == "skill-conflict"
            finally:
                fcntl.flock(blocker_fd, fcntl.LOCK_UN)
                os.close(blocker_fd)


# ---------------------------------------------------------------------------
# Concurrent subprocess test (acceptance metric §6)
# ---------------------------------------------------------------------------

_WORKER_SCRIPT = textwrap.dedent(
    """\
    import fcntl
    import json
    import os
    import sys
    import time
    from pathlib import Path

    project_root = sys.argv[1]
    hermes_home = sys.argv[2]
    skill_cat = sys.argv[3]
    skill_name = sys.argv[4]
    state_file = sys.argv[5]
    worker_id = sys.argv[6]

    sys.path.insert(0, project_root)
    os.environ["HERMES_HOME"] = hermes_home

    from dgmh.locks import acquire_skill_lock, SkillLockError

    try:
        with acquire_skill_lock(skill_cat, skill_name, timeout_s=10.0, retries=5):
            # Read current counter
            try:
                data = json.loads(Path(state_file).read_text())
            except Exception:
                data = {"counter": 0, "writers": []}

            # Simulate work
            time.sleep(0.05)

            # Increment counter
            data["counter"] += 1
            data["writers"].append(worker_id)

            # Write back
            import tempfile
            fd, tmp = tempfile.mkstemp(dir=str(Path(state_file).parent))
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
            os.replace(tmp, state_file)

        sys.exit(0)
    except SkillLockError:
        sys.exit(2)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    """
)


class TestConcurrentSubprocessLock:
    def test_two_processes_consistent_state(self, tmp_path):
        """Two subprocesses competing for same lock leave state consistent.

        Acceptance metric §6 (v2 addendum): second writer either waits or
        exits cleanly; state file is never torn.
        """
        hermes_home = str(tmp_path)
        state_file = tmp_path / "state.json"
        state_file.write_text('{"counter": 0, "writers": []}', encoding="utf-8")

        # Write worker script to tmp file
        worker_py = tmp_path / "worker.py"
        worker_py.write_text(_WORKER_SCRIPT, encoding="utf-8")

        procs = []
        for i in range(2):
            p = subprocess.Popen(
                [
                    sys.executable,
                    str(worker_py),
                    str(PROJECT_ROOT),
                    hermes_home,
                    "cat-a",
                    "concurrent-skill",
                    str(state_file),
                    str(i),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            procs.append(p)

        results = []
        for p in procs:
            stdout, stderr = p.communicate(timeout=30)
            results.append((p.returncode, stdout.decode(), stderr.decode()))

        # All processes must exit cleanly (0) or with SkillLockError (2)
        for rc, out, err in results:
            assert rc in (0, 2), f"Worker crashed with rc={rc}, stderr={err!r}"

        # State file must be valid JSON (not torn)
        data = json.loads(state_file.read_text())
        assert isinstance(data["counter"], int)
        assert isinstance(data["writers"], list)

        # Successful writers incremented counter once each
        successful = sum(1 for rc, _, _ in results if rc == 0)
        assert data["counter"] == successful, (
            f"counter={data['counter']} should equal {successful} successful writers"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def patch_hermes_home(tmp_path: Path):
    """Redirect HERMES_HOME for the duration of the test."""
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        yield
