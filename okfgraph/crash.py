"""Crash reports: full tracebacks for unexpected failures, on disk.

The CLI turns unclassified exceptions into a one-line ``[ERROR] INTERNAL``
(§4), and a failure while importing okfgraph itself never reaches that
handler at all. Either way, a harness that keeps only the last lines of
stderr loses the frames that name the real cause. Every such failure
therefore also writes the complete traceback to a crash file and names
that file on the final ``[ERROR]`` line, which survives any ``tail``.

Stdlib only, no okfgraph imports: the start-up guard must work while the
rest of the package fails to import.
"""
from __future__ import annotations

import datetime as _dt
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Optional

KEEP = 20  # newest crash files kept; older ones are pruned


def crash_dir() -> Path:
    """``OKF_CRASH_DIR``, else the per-user state dir + ``okfgraph/crash``.

    Windows: ``%LOCALAPPDATA%``; elsewhere ``$XDG_STATE_HOME`` or
    ``~/.local/state``.
    """
    explicit = os.getenv("OKF_CRASH_DIR")
    if explicit:
        return Path(explicit)
    if os.name == "nt" and os.getenv("LOCALAPPDATA"):
        base = Path(os.environ["LOCALAPPDATA"])
    else:
        base = Path(os.getenv("XDG_STATE_HOME")
                    or Path.home() / ".local" / "state")
    return base / "okfgraph" / "crash"


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("okfgraph")
    except Exception:
        return "unknown"


def write_crash_report(exc: BaseException, *, command: str = "") -> Optional[str]:
    """Write ``exc``'s full traceback plus context; return the file path.

    Never raises: an unwritable crash dir returns None and the caller
    falls back to the console traceback alone.
    """
    try:
        d = crash_dir()
        d.mkdir(parents=True, exist_ok=True)
        now = _dt.datetime.now()
        # Microseconds: two crashes in one shell session never collide.
        name = f"okf-{now:%Y%m%d-%H%M%S-%f}-{os.getpid()}.log"
        path = d / name
        tb = "".join(traceback.format_exception(type(exc), exc,
                                                exc.__traceback__))
        path.write_text(
            f"time:     {now.isoformat()}\n"
            f"command:  {command or '(start-up)'}\n"
            f"argv:     {sys.argv!r}\n"
            f"okfgraph: {_version()}\n"
            f"python:   {sys.version.split()[0]} ({sys.executable})\n"
            f"platform: {platform.platform()}\n"
            f"cwd:      {os.getcwd()}\n\n{tb}",
            encoding="utf-8",
        )
        _prune(d)
        return str(path)
    except Exception:
        return None


def _prune(d: Path) -> None:
    try:
        files = sorted(d.glob("okf-*.log"), key=lambda p: p.stat().st_mtime)
        for old in files[:-KEEP]:
            old.unlink()
    except OSError:
        pass
