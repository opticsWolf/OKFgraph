"""Hermetic working directory for every test (0.10.1 quirk fix).

``okfgraph.toml`` is looked up CWD-first, so a stray TOML in the
invocation directory (e.g. the repo-local ``db_path = "kb.db"`` pin)
silently re-pins the database and shadows test fixtures — this exact
interaction failed ``test_toml_roots_relative_to_toml`` whenever pytest
ran from the repo root. Each test therefore runs in an empty tmp dir;
tests that need their own CWD keep overriding it via ``monkeypatch.chdir``
or an explicit subprocess ``cwd`` (including the test that pins the
CWD-first precedence itself). ``PYTHONPATH`` gains the repo root so that
``sys.executable -m okfgraph.cli`` subprocesses — which inherit CWD, not
``sys.path`` — still resolve the working tree under test.

Teardown is manual (``try/finally``), not via the ``monkeypatch``
fixture: depending on ``monkeypatch`` would hoist its setup ahead of
module-level autouse fixtures and invert the teardown order those
fixtures rely on (seen with ``test_gpu_hint``'s cache-clearing fixture).
"""
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path):
    prev_cwd = os.getcwd()
    os.chdir(tmp_path)
    prev_pythonpath = os.environ.get("PYTHONPATH")
    entries = prev_pythonpath.split(os.pathsep) if prev_pythonpath else []
    if str(REPO_ROOT) not in entries:
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [str(REPO_ROOT), *entries])
    try:
        yield
    finally:
        os.chdir(prev_cwd)
        if prev_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = prev_pythonpath
