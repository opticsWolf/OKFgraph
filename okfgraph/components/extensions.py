"""Offline Ladybug extensions: find, fetch, copy and verify (0.11.0, O5).

Hybrid search needs Ladybug's ``vector`` and ``fts`` extensions. Ladybug
downloads them on first ``INSTALL`` from ``extension.ladybugdb.com`` and
caches them under ``~/.lbdb/extension/<version>/<platform>/``; an offline
machine therefore lost hybrid search silently (a log warning, then
``SEARCH_UNAVAILABLE`` on every search).

``okf setup extensions`` fills one directory with both files — copied from
``--from DIR``, from Ladybug's own cache, or (online) downloaded once — and
verifies they load. ``[database] extension_dir`` then makes every router
load them by path and never download.
"""
from __future__ import annotations

import os
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

EXTENSIONS = ("vector", "fts")


def ladybug_version() -> str:
    import ladybug
    return getattr(ladybug, "__version__", None) or _dist_version()


def _dist_version() -> str:
    from importlib.metadata import version
    return version("ladybug")


def extension_abi_version(version: Optional[str] = None) -> str:
    """The version directory Ladybug uses for extensions (``X.Y.0``)."""
    parts = (version or ladybug_version()).split(".")
    return f"{parts[0]}.{parts[1]}.0"


def platform_tag() -> str:
    """Ladybug's platform directory name (``linux_amd64``, ``osx_arm64``, ...)."""
    system = {"Darwin": "osx", "Windows": "win"}.get(platform.system(), "linux")
    machine = platform.machine().lower()
    arch = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64",
            "arm64": "arm64"}.get(machine, machine)
    return f"{system}_{arch}"


def ladybug_cache_dir() -> Path:
    """Where Ladybug itself caches downloaded extensions."""
    return (Path.home() / ".lbdb" / "extension" / extension_abi_version()
            / platform_tag())


def default_target_dir() -> Path:
    """``$XDG_DATA_HOME/okfgraph/extensions/<abi>/<platform>`` (or ~/.local/share)."""
    base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "okfgraph" / "extensions" / extension_abi_version() / platform_tag()


def find_extension(directory: Path, ext: str) -> Optional[Path]:
    """``lib<ext>.lbug_extension`` in ``directory`` — flat or ``<ext>/`` layout."""
    name = f"lib{ext}.lbug_extension"
    for cand in (directory / name, directory / ext / name):
        if cand.is_file():
            return cand
    return None


def _download_into_cache(ext: str) -> None:
    """Let Ladybug download ``ext`` into its own cache (needs network)."""
    import ladybug as lb
    with tempfile.TemporaryDirectory() as tmp:
        db = lb.Database(str(Path(tmp) / "probe.db"))
        conn = lb.Connection(db)
        try:
            conn.execute(f"INSTALL {ext};")
        finally:
            conn.close()
            db.close()


def verify(directory: Path) -> Dict[str, str]:
    """Load every extension from ``directory`` in a scratch database.

    Returns ``{ext: "ok" | error message}``.
    """
    import ladybug as lb
    out: Dict[str, str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        db = lb.Database(str(Path(tmp) / "verify.db"))
        conn = lb.Connection(db)
        try:
            for ext in EXTENSIONS:
                path = find_extension(directory, ext)
                if path is None:
                    out[ext] = "missing"
                    continue
                try:
                    conn.execute(f"LOAD EXTENSION '{path.as_posix()}'")
                    out[ext] = "ok"
                except Exception as exc:
                    out[ext] = str(exc)
        finally:
            conn.close()
            db.close()
    return out


def setup_extensions(target: Optional[Path] = None, source: Optional[Path] = None,
                     download: bool = True) -> Dict[str, object]:
    """Fill ``target`` with both extensions and verify they load.

    Order per extension: ``source`` (if given), then Ladybug's cache, then
    (``download=True``) a one-off ``INSTALL`` into that cache. Raises
    ``FileNotFoundError`` naming what is missing when none of these has it.
    Returns ``{"target", "copied": {ext: from}, "verified", "toml"}``.
    """
    target = Path(target) if target else default_target_dir()
    target.mkdir(parents=True, exist_ok=True)
    copied: Dict[str, str] = {}
    missing: List[str] = []
    for ext in EXTENSIONS:
        found = find_extension(Path(source), ext) if source else None
        if found is None and source is None:
            found = find_extension(ladybug_cache_dir(), ext)
            if found is None and download:
                try:
                    _download_into_cache(ext)
                except Exception:
                    pass
                found = find_extension(ladybug_cache_dir(), ext)
        if found is None:
            missing.append(ext)
            continue
        dest = target / f"lib{ext}.lbug_extension"
        if found.resolve() != dest.resolve():
            shutil.copy2(found, dest)
        copied[ext] = str(found)
    if missing:
        where = (f"--from {source}" if source else
                 f"Ladybug's cache {ladybug_cache_dir()}"
                 + ("" if download else " (download disabled)"))
        raise FileNotFoundError(
            f"extension(s) {', '.join(missing)} not found in {where}; copy "
            f"lib<ext>.lbug_extension for Ladybug {extension_abi_version()} "
            f"{platform_tag()} from an online machine's {ladybug_cache_dir()}")
    checks = verify(target)
    return {
        "target": str(target),
        "copied": copied,
        "verified": checks,
        "ok": all(v == "ok" for v in checks.values()),
        "toml": f'[database]\nextension_dir = "{target.as_posix()}"',
    }
