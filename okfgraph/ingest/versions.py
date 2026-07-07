"""Known-good RapidAI package versions and runtime version checking.

Gap #15 — RapidAI Version Pinning
==================================

RapidAI packages move fast. Breaking API changes between minor versions
can silently break the ingestion pipeline. This module provides:

1. **Version pins** — exact versions in ``pyproject.toml`` [project.optional-dependencies].pdf-ingest
2. **Runtime warning** — checks installed versions against the known-good list
   on first import of the ``okfgraph.ingest`` package and logs a warning
   if any package is outside the allowed range.

Usage
-----
The check runs automatically when ``okfgraph.ingest`` is imported.
To silence the warning, set the environment variable
``OKFGRAPH_INGEST_ALLOW_UNPINNED=1`` or call
:func:`check_rapid_versions` manually with ``warn=False``.
"""

from __future__ import annotations

import logging
import os
from importlib.metadata import version as _pkg_version
from typing import Dict, Optional, Tuple

log = logging.getLogger(__name__)

# ── Known-good version ranges ──────────────────────────────────────────────
# Each entry: (package_name, exact_version_or_range, display_name)
# The "exact" field is the version that has been tested and verified.
# We allow ±1 patch version as a tolerance band.

_KNOWN_GOOD: Dict[str, Tuple[str, str]] = {
    "rapidocr": ("1.5.2", "rapidocr"),
    "rapid_latex_ocr": ("1.0.13", "rapid_latex_ocr"),
    "rapid_layout": ("0.2.0", "rapid_layout"),
    "rapid_table": ("1.0.3", "rapid_table"),
    "pdf_oxide": ("0.2.1", "pdf_oxide"),
}


def _parse_version(v: str) -> Tuple[int, ...]:
    """Parse a version string like '1.5.2' into a tuple of ints."""
    parts: list[int] = []
    for part in v.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            break
    # Pad to at least 3 components
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def _is_within_tolerance(installed: str, known: str, tolerance: int = 1) -> bool:
    """Check if installed version is within tolerance of the known-good version.

    Tolerance is in patch-level steps. For example, with tolerance=1:
    - known=1.5.2 → accepts 1.5.1, 1.5.2, 1.5.3
    - known=1.5.0 → accepts 1.4.9 (if minor matches), 1.5.x, 1.6.0 (if minor ±1)

    We use a simple heuristic: same major, minor within ±1, patch within tolerance.
    """
    installed_tuple = _parse_version(installed)
    known_tuple = _parse_version(known)

    # Same major version required
    if installed_tuple[0] != known_tuple[0]:
        return False

    # Minor version within ±1
    if abs(installed_tuple[1] - known_tuple[1]) > 1:
        return False

    # Patch version within tolerance (only if minor matches)
    if installed_tuple[1] == known_tuple[1]:
        return abs(installed_tuple[2] - known_tuple[2]) <= tolerance

    # Different minor — still allow if within ±1 minor
    return True


def check_rapid_versions(warn: bool = True, env_allow: str = "OKFGRAPH_INGEST_ALLOW_UNPINNED") -> list[str]:
    """Check installed RapidAI package versions against known-good list.

    Returns a list of warnings (empty if all packages are within tolerance).

    Parameters
    ----------
    warn : bool
        If True, emit a logging warning for each out-of-range package.
    env_allow : str
        Environment variable name that, if set to a truthy value, silences
        all warnings.

    Returns
    -------
    list[str]
        List of warning messages (empty if everything is fine).
    """
    # Check environment variable first
    if os.environ.get(env_allow, "").lower() in ("1", "true", "yes", "on"):
        return []

    warnings: list[str] = []

    for pkg_key, (known_ver, display_name) in _KNOWN_GOOD.items():
        try:
            installed_ver = _pkg_version(pkg_key)
        except Exception:
            # Package not installed — that's fine, it's optional
            continue

        if not _is_within_tolerance(installed_ver, known_ver):
            msg = (
                f"okfgraph.ingest: {display_name} {installed_ver} is installed "
                f"(known-good: {known_ver}). "
                f"Untested versions may break the ingestion pipeline. "
                f"Install the pinned version: pip install {pkg_key}=={known_ver} "
                f"(or set {env_allow}=1 to silence this warning)."
            )
            warnings.append(msg)
            if warn:
                log.warning(msg)

    return warnings


# ── Auto-run on import ─────────────────────────────────────────────────────
# This runs once when the package is first imported.
# Suppress with OKFGRAPH_INGEST_ALLOW_UNPINNED=1.
check_rapid_versions(warn=True)
