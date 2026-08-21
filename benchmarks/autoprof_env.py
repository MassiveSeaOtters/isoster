"""Canonical location of the isolated AutoProf interpreter.

AutoProf 1.3.4 pins ``numpy<2`` and ``photutils<=1.5``, so it cannot share
the project's ``uv`` environment. It lives in its own virtual environment
and is driven over subprocess. Two benchmark trees need to find that
interpreter -- the standalone adapter in ``benchmarks/utils`` and the
campaign fitter in ``benchmarks/exhausted`` -- and they used to carry
separate hardcoded defaults that had drifted apart and both gone stale.
The path is defined once here instead.

The install recipe lives in ``benchmarks/exhausted/README.md``.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Default install location, matching the recipe in
#: ``benchmarks/exhausted/README.md`` and every campaign YAML. Deliberately
#: not under ``/tmp``: macOS prunes that directory and has been observed to
#: strip individual ``.py`` files out of a venv mid-campaign.
DEFAULT_AUTOPROF_VENV_PYTHON = "~/.venvs/autoprof_venv/bin/python"

#: Environment variable that overrides the default for a whole shell session.
AUTOPROF_PYTHON_ENV_VAR = "AUTOPROF_PYTHON"


def resolve_autoprof_python(candidate: str | None = None) -> str:
    """Return an absolute path to the AutoProf interpreter.

    Resolution order: the explicit ``candidate`` argument, then the
    ``AUTOPROF_PYTHON`` environment variable, then
    :data:`DEFAULT_AUTOPROF_VENV_PYTHON`. ``~`` is expanded at every step,
    because the value is handed to :func:`subprocess.run` without a shell
    and so would otherwise be taken literally.

    The path is returned whether or not it exists; callers probe it and
    report a missing interpreter as a skip rather than a failure.
    """
    raw = candidate or os.environ.get(AUTOPROF_PYTHON_ENV_VAR) or DEFAULT_AUTOPROF_VENV_PYTHON
    return str(Path(raw).expanduser())


def autoprof_install_hint(resolved_path: str | os.PathLike[str]) -> str:
    """Return the recipe to print when the interpreter is missing."""
    venv_root = Path(resolved_path).parents[1]
    return (
        f"autoprof venv python not found: {resolved_path}. Create it with "
        f"`uv venv --python 3.10 {venv_root}` and "
        f"`uv pip install --python {resolved_path} 'autoprof==1.3.4'`, "
        f"then re-run. Full recipe: benchmarks/exhausted/README.md."
    )
