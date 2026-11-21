#!/usr/bin/env python3
"""Resolve the frozen Python 0.15 oracle from its git tag.

The parity harness must not depend on the old ``okfgraph/`` package in the
working tree: that package is removed at cutover. All oracle imports therefore
come from a detached worktree of ``python-oracle-0.15.0`` unless an explicit
local override is supplied.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAG = "python-oracle-0.15.0"
DEFAULT_WORKTREE = ROOT / "parity/tmp/oracle-0.15.0"


def _valid(path: Path) -> bool:
    return (path / "okfgraph" / "__init__.py").is_file()


def oracle_root(
    override: str | Path | None = None,
    *,
    tag: str | None = None,
) -> Path:
    """Return a checkout of the oracle tag, creating a cached worktree if needed."""
    raw = override or os.environ.get("OKFGRAPH_ORACLE_ROOT")
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        if not _valid(path):
            raise SystemExit(f"OKFGRAPH_ORACLE_ROOT is not a Python oracle checkout: {path}")
        return path

    tag_name = tag or os.environ.get("OKFGRAPH_ORACLE_TAG") or DEFAULT_TAG
    worktree = DEFAULT_WORKTREE.resolve()

    if worktree.exists() and _valid(worktree):
        return worktree
    if worktree.exists():
        subprocess.run(
            ["git", "worktree", "prune"], cwd=ROOT, check=False,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        shutil.rmtree(worktree, ignore_errors=True)

    worktree.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "worktree", "prune"], cwd=ROOT, check=False)
    result = subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree), tag_name],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode or not _valid(worktree):
        raise SystemExit(
            f"cannot check out oracle tag {tag_name}: {result.stderr.strip()}"
        )
    return worktree


def prepend_oracle_path(
    override: str | Path | None = None,
    *,
    tag: str | None = None,
) -> Path:
    """Resolve the oracle and put it first on ``sys.path``."""
    path = oracle_root(override, tag=tag)
    root = str(path)
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    return path
