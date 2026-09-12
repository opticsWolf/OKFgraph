"""Shared link primitives: extraction + Obsidian-style name resolution.

All three import paths (bundle batch, single concept, single upsert) and the
structural diff resolve links through these pure helpers so path-links and
``[[wikilinks]]`` behave identically everywhere. No database access here —
callers supply the known-id set / name index.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

MD_LINK_RE = re.compile(r"\[.*?\]\((.*?\.md)\)")
WIKI_RE = re.compile(r"\[\[(.*?)\]\]")
SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def extract_md_links(body: str) -> List[str]:
    """Raw targets of ``[text](target.md)`` links (external URLs included)."""
    return MD_LINK_RE.findall(body or "")


def extract_wikilinks(body: str) -> List[str]:
    """Raw targets of ``[[target]]`` / ``[[target|display]]`` (display stripped)."""
    return [
        m.split("|", 1)[0].strip()
        for m in WIKI_RE.findall(body or "")
    ]


def is_external(raw: str) -> bool:
    """True for ``scheme:...`` targets (http, mailto, okf-asset, ...)."""
    return bool(SCHEME_RE.match((raw.split("#", 1)[0]).strip()))


def normalize_path_link(raw: str) -> str:
    """Map a markdown link target to a concept id (path without extension)."""
    return raw.lstrip("./").replace("\\", "/").replace(".md", "").lstrip("/")


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return [str(v) for v in value]
    except TypeError:  # pragma: no cover - defensive
        return [str(value)]


def build_name_index(
    concepts: Iterable[Mapping[str, Any]],
) -> Tuple[Dict[str, Dict[str, str]], Set[str]]:
    """Build the wikilink name index.

    Each concept mapping needs ``id`` plus any of ``uid`` (frontmatter
    ``id:``, preserved as ``uid`` on the node), ``aliases``/``alias``,
    ``title``. Keys are lowercased; precedence at resolution is
    uid → alias → title → filename-stem.

    Returns ``(maps, ambiguous)`` where ``maps`` is
    ``{"uid": {...}, "alias": {...}, "title": {...}, "stem": {...}}`` and
    ``ambiguous`` holds every key claimed by more than one concept.
    Ambiguous names never resolve (deterministic miss, never a guess).
    """
    maps: Dict[str, Dict[str, str]] = {
        "uid": {}, "alias": {}, "title": {}, "stem": {},
    }
    claimed: Dict[str, Dict[str, str]] = {
        "uid": {}, "alias": {}, "title": {}, "stem": {},
    }

    def _add(kind: str, key: str, cid: str) -> None:
        key = (key or "").strip().lower()
        if not key:
            return
        if key in maps[kind] and maps[kind][key] != cid:
            claimed[kind][key] = cid
        else:
            maps[kind].setdefault(key, cid)

    for c in concepts:
        cid = c.get("id", "")
        if not cid:
            continue
        uid = c.get("uid")
        if uid is not None:
            _add("uid", str(uid), cid)
        for a in _as_list(c.get("aliases")) + _as_list(c.get("alias")):
            _add("alias", a, cid)
        title = c.get("title")
        if title:
            _add("title", str(title), cid)
        stem = cid.rsplit("/", 1)[-1]
        _add("stem", stem, cid)

    ambiguous: Set[str] = set()
    for kind in maps:
        for key in claimed[kind]:
            maps[kind].pop(key, None)
            ambiguous.add(f"{kind}:{key}")
    return maps, ambiguous


def resolve_wiki(
    reference: str,
    maps: Mapping[str, Mapping[str, str]],
    known_ids: Set[str],
) -> Optional[str]:
    """Resolve a wikilink reference to a concept id, or None.

    Order: exact concept id → id-minus-``.md`` → uid → alias → title → stem
    (all case-insensitive except the exact-id probe). Fragment (``#sec``) and
    surrounding whitespace are ignored.
    """
    ref = (reference or "").split("#", 1)[0].strip()
    if not ref:
        return None
    if ref in known_ids:
        return ref
    cand = ref[:-3] if ref.lower().endswith(".md") else ref
    if cand in known_ids:
        return cand
    low = ref.lower()
    if low.endswith(".md"):
        low = low[:-3]
    for kind in ("uid", "alias", "title", "stem"):
        hit = maps.get(kind, {}).get(low)
        if hit is not None:
            return hit
    return None
