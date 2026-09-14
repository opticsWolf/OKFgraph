"""Multi-root bundle helpers (0.4.0, Phase 2 §2.1).

One graph, N live roots, no copies, no ID collisions. The constructor
``bundle_root`` stays the primary tree (its files keep bare IDs — the
empty-alias rule, so legacy single-root graphs need no migration);
``roots`` maps additional ``alias → path`` pairs whose files get
``@alias/rel`` concept IDs.

Derivation lives in ONE function: ``parse_source_file(..., alias=...)``
(:mod:`okfgraph.components.import_`); this module owns validation,
prefix math, and longest-prefix root resolution shared by import, delta,
links, diff, and ingest.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

#: Alias charset (Phase 2 §2.1): filename-safe, and excluding ``:`` so an
#: alias can never resemble a URI scheme inside a link target. ``@`` is
#: excluded (it's the namespace sigil) as are ``.``/``/`` (path safety).
ALIAS_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def validate_roots(
    roots: Optional[Mapping[str, str]],
    primary: Optional[str] = None,
) -> Dict[str, Path]:
    """Validate an ``{alias: path}`` mapping into ``{alias: resolved Path}``.

    ``None``/``{}`` → ``{}`` (legacy single-root mode). When ``primary``
    (the constructor ``bundle_root``) is given it joins the overlap check
    as the ``""``-alias root: a file living in two roots would mint two
    identities, so equality and nesting in either direction are rejected
    fail-fast. Existence is NOT required — an absent root is a legal
    liveness state (unmounted ≠ deleted, §2.3), validated by shape only.
    """
    out: Dict[str, Path] = {}
    if not roots:
        return out
    for alias, path in roots.items():
        if not isinstance(alias, str) or not alias:
            raise ValueError(f"root alias must be a non-empty string, got {alias!r}")
        if alias.startswith("@"):
            raise ValueError(
                f"root alias {alias!r} must not start with '@' (namespace sigil)"
            )
        if not ALIAS_RE.match(alias):
            raise ValueError(
                f"root alias {alias!r} must match {ALIAS_RE.pattern} "
                "(letters/digits/_/-, no ':' so it can't mimic a URI scheme)"
            )
        if not isinstance(path, (str, Path)) or not str(path):
            raise ValueError(f"root {alias!r} needs a non-empty path")
        if alias in out:
            raise ValueError(f"duplicate root alias {alias!r}")
        out[alias] = Path(path).resolve()
    paths: Dict[str, Path] = dict(out)
    if primary is not None:
        paths[""] = Path(primary).resolve()
    names = list(paths)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = paths[names[i]], paths[names[j]]
            if a == b or a in b.parents or b in a.parents:
                raise ValueError(
                    f"roots {names[i]!r} ({paths[names[i]]}) and "
                    f"{names[j]!r} ({paths[names[j]]}) overlap: a file in "
                    "two roots would mint two identities"
                )
    return out


def resolve_alias_for_path(
    path: Path, roots: Mapping[str, Path]
) -> Optional[str]:
    """Longest-prefix-match of ``path`` against ``roots`` (None = outside).

    Used for path-based writes (single-file import, md ingest): a file
    inside a root mints that root's namespaced ID; anything else keeps
    the legacy bare-stem fallback.
    """
    if not roots:
        return None
    try:
        rp = Path(path).resolve()
    except OSError:
        return None
    best: Optional[str] = None
    best_len = -1
    for alias, root in roots.items():
        if rp == root or root in rp.parents:
            n = len(root.parts)
            if n > best_len:
                best, best_len = alias, n
    return best


def namespaced_id(alias: str, native_id: str) -> str:
    """``(a, sub/doc)`` → ``@a/sub/doc``; empty alias keeps the bare ID."""
    return f"@{alias}/{native_id}" if alias else native_id


def qualify_alias_link(raw: str, aliases) -> str:
    """Rewrite ``alias/rest`` → ``@alias/rest`` for a known alias.

    Pure helper shared by import link resolution and diff snapshots:
    the alias segment matches case-insensitively (canonical casing
    restored), the rest stays exact, ``#fragment`` preserved. The `@`-form
    passes through (it hits the exact-id probe downstream).
    """
    if not aliases or "/" not in raw:
        return raw
    ref, hash_, frag = raw.partition("#")
    first, sep, rest = ref.strip().partition("/")
    if not sep or not rest.strip():
        return raw
    seg = first[1:] if first.startswith("@") else first
    for alias in aliases:
        if seg.lower() == alias.lower():
            return f"@{alias}/{rest.strip()}{hash_}{frag}"
    return raw


def split_namespaced(concept_id: str) -> Tuple[str, str]:
    """``@a/b/c`` → ``(a, b/c)``; anything else → ``('', id)``."""
    if concept_id.startswith("@") and "/" in concept_id:
        alias, _, rest = concept_id[1:].partition("/")
        if rest and ALIAS_RE.match(alias):
            return alias, rest
    return "", concept_id


def prefix_key(alias: str, native_rel: str) -> str:
    """Delta key space: FileHash/DirHash/DeletedPath paths are DB keys, so
    each root gets its own namespace (``@a/rel.md``); empty alias is the
    legacy bare key. Idempotent — never double-prefixes."""
    if not alias or native_rel.startswith(f"@{alias}/"):
        return native_rel
    return f"@{alias}/{native_rel}"


def strip_prefix(alias: str, key: str) -> Optional[str]:
    """Inverse of :func:`prefix_key`: None when ``key`` is another root's."""
    if not alias:
        # Single-root mode owns the whole table, including '@'-prefixed
        # rows from re-imported multi-root exports (empty-alias rule).
        return key
    pre = f"@{alias}/"
    return key[len(pre):] if key.startswith(pre) else None
