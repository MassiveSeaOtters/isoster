"""Shared benchmark/profiling metadata utilities."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import scipy


def get_git_sha(project_root: Optional[Path] = None) -> str:
    """Return the current git SHA for metadata recording."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            capture_output=True,
            check=True,
        )
    except Exception:
        return "unknown"

    return result.stdout.strip()


def get_git_worktree_state(project_root: Optional[Path] = None) -> Dict[str, object]:
    """Describe whether the tree differs from the recorded commit.

    A ``git_sha`` alone is misleading provenance: a benchmark run from a
    modified working tree records the SHA of a commit that does not contain the
    code that ran, and cannot be reconstructed from it. This reports whether
    the tree was dirty and, if so, a hash of the diff plus the changed paths,
    so an archive is at least honest about being unreproducible.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    def _git(*args: str) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=project_root,
                text=True,
                capture_output=True,
                check=True,
            )
        except Exception:
            return None
        return result.stdout

    status = _git("status", "--porcelain")
    if status is None:
        return {"dirty": None, "note": "git unavailable; provenance unknown"}

    changed = [line[3:].strip() for line in status.splitlines() if line.strip()]
    if not changed:
        return {"dirty": False}

    diff = _git("diff", "HEAD") or ""
    untracked = [line[3:].strip() for line in status.splitlines() if line.startswith("??")]
    return {
        "dirty": True,
        "changed_paths": sorted(changed),
        "untracked_paths": sorted(untracked),
        # Named for what it actually covers. ``git diff HEAD`` shows modifications
        # to *tracked* files only, so a tree dirty solely through untracked files
        # hashes an empty diff. The untracked paths are listed above, but their
        # contents are not hashed -- do not read this digest as identifying the
        # full working state.
        "tracked_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "note": (
            "This run did NOT come from the recorded commit: the working tree "
            "carried uncommitted changes. The SHA identifies the parent commit "
            "only. tracked_diff_sha256 covers modifications to tracked files; "
            "untracked file contents are not hashed, so an identical digest "
            "does not prove an identical tree. Re-run from a clean tree before "
            "treating these numbers as reproducible provenance."
        ),
    }


def _optional_module_version(module_name: str) -> Optional[str]:
    """Return module version if import succeeds, else None."""
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def collect_environment_metadata(
    project_root: Optional[Path] = None,
    extra_env_keys: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    """Collect standard machine/environment metadata for benchmark artifacts."""
    keys = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_DISABLE_JIT"]
    if extra_env_keys is not None:
        keys.extend(list(extra_env_keys))

    unique_keys = sorted(set(keys))
    selected_environment = {key: os.getenv(key, "") for key in unique_keys}

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_sha(project_root=project_root),
        "git_worktree": get_git_worktree_state(project_root=project_root),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": _optional_module_version("numba"),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "environment_variables": selected_environment,
    }


def write_json(path: Path, payload: Dict[str, object]) -> None:
    """Write JSON payload with deterministic key ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_pointer:
        json.dump(payload, file_pointer, indent=2, sort_keys=True)
