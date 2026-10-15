"""Shared link primitives: extraction + Obsidian-style name resolution.

All three import paths (bundle batch, single concept, single upsert) and the
structural diff resolve links through these pure helpers so path-links and
``[[wikilinks]]`` behave identically everywhere. No database access here —
callers supply the known-id set / name index.
"""

from __future__ import annotations

import posixpath
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
    """Map a markdown link target to a concept id (path without extension).

    Legacy, root-relative reading: leading ``./``/``../``/``/`` are stripped,
    so ``../ev/t.md`` from anywhere means ``ev/t``. Kept as the fallback of
    :func:`resolve_path_link`, which tries the source-relative reading first.
    """
    return raw.lstrip("./").replace("\\", "/").replace(".md", "").lstrip("/")


def relative_path_link(raw: str, source_id: str) -> Optional[str]:
    """The source-relative reading of a markdown link (0.11.0, O1).

    ``[t](t.md)`` written in ``findings/a.md`` means ``findings/t`` — the
    way every Markdown renderer resolves it. Returns ``None`` for absolute
    (``/x.md``) targets and for targets that climb above the bundle root,
    where only the legacy root-relative reading applies.
    """
    target = (raw or "").split("#", 1)[0].strip().replace("\\", "/")
    if not target or target.startswith("/"):
        return None
    if target.lower().endswith(".md"):
        target = target[:-3]
    base = posixpath.dirname(source_id)
    joined = posixpath.normpath(posixpath.join(base, target)) if base else posixpath.normpath(target)
    if joined.startswith("..") or joined in (".", ""):
        return None
    return joined


def resolve_path_link(
    raw: str, source_id: str, known_ids: Set[str],
    path_ids: Optional[Mapping[str, str]] = None,
) -> Tuple[Optional[str], str]:
    """Resolve a markdown path link to ``(target_or_None, recorded_target)``.

    Order: the source-relative reading, then the legacy root-relative one
    (:func:`normalize_path_link`), so bundles written against the old rule
    keep resolving. ``recorded_target`` is what a miss records on its
    ``BrokenLink``: the source-relative reading when there is one (the
    file a later import is most likely to create), else the legacy form.

    ``path_ids`` maps a file path (without extension) to the concept id of
    the file there, for bundles where the two differ (frontmatter
    identity, O3): a Markdown link names a *file*, so it resolves to that
    file's concept.
    """
    rel = relative_path_link(raw, source_id)
    legacy = normalize_path_link(raw.split("#", 1)[0])
    for cand in (rel, legacy):
        if not cand:
            continue
        if cand in known_ids:
            return cand, cand
        if path_ids and cand in path_ids:
            return path_ids[cand], path_ids[cand]
    return None, rel or legacy


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
    """See :func:`build_name_index_full`; returns ``(maps, ambiguous)``."""
    maps, ambiguous, _candidates = build_name_index_full(concepts)
    return maps, ambiguous


def build_name_index_full(
    concepts: Iterable[Mapping[str, Any]],
) -> Tuple[Dict[str, Dict[str, str]], Set[str], Dict[str, Dict[str, List[str]]]]:
    """Build the wikilink name index.

    Each concept mapping needs ``id`` plus any of ``uid`` (frontmatter
    ``id:``, preserved as ``uid`` on the node), ``aliases``/``alias``,
    ``title``. Keys are lowercased; precedence at resolution is
    uid → alias → title → filename-stem.

    Returns ``(maps, ambiguous, candidates)`` where ``maps`` is
    ``{"uid": {...}, "alias": {...}, "title": {...}, "stem": {...}}``,
    ``ambiguous`` holds every key claimed by more than one concept, and
    ``candidates[kind][key]`` lists the claiming ids (0.11.0, O12) so a
    miss can say *why* it missed. Ambiguous names never resolve
    (deterministic miss, never a guess).
    """
    maps: Dict[str, Dict[str, str]] = {
        "uid": {}, "alias": {}, "title": {}, "stem": {},
    }
    claimed: Dict[str, Dict[str, List[str]]] = {
        "uid": {}, "alias": {}, "title": {}, "stem": {},
    }

    def _add(kind: str, key: str, cid: str) -> None:
        key = (key or "").strip().lower()
        if not key:
            return
        if key in maps[kind] and maps[kind][key] != cid:
            ids = claimed[kind].setdefault(key, [maps[kind][key]])
            if cid not in ids:
                ids.append(cid)
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
    candidates: Dict[str, Dict[str, List[str]]] = {k: {} for k in maps}
    for kind in maps:
        for key, ids in claimed[kind].items():
            maps[kind].pop(key, None)
            ambiguous.add(f"{kind}:{key}")
            candidates[kind][key] = sorted(ids)
    return maps, ambiguous, candidates


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


def resolve_wiki_detail(
    reference: str,
    maps: Mapping[str, Mapping[str, str]],
    known_ids: Set[str],
    candidates: Mapping[str, Mapping[str, List[str]]],
) -> Tuple[Optional[str], Optional[str], List[str]]:
    """:func:`resolve_wiki` plus the reason for a miss (0.11.0, O12).

    Returns ``(target, reason, candidate_ids)``: ``reason`` is ``None`` on
    a hit, ``"ambiguous"`` when the name is claimed by several concepts at
    the first tier that knows it (``candidate_ids`` then lists them), and
    ``"missing"`` otherwise.
    """
    hit = resolve_wiki(reference, maps, known_ids)
    if hit is not None:
        return hit, None, []
    ref = (reference or "").split("#", 1)[0].strip().lower()
    if ref.endswith(".md"):
        ref = ref[:-3]
    for kind in ("uid", "alias", "title", "stem"):
        ids = candidates.get(kind, {}).get(ref)
        if ids:
            return None, "ambiguous", list(ids)
    return None, "missing", []


#: Relation kinds frontmatter may declare (0.11.0, O6). ``mention`` is the
#: kind every body link carries and cannot be declared.
RELATION_KIND_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
MENTION = "mention"


def frontmatter_relations(value: Any) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Normalise a ``relations:`` mapping to ``([(kind, target)], problems)``.

    Accepts ``{kind: target}`` or ``{kind: [targets]}``. Kinds must match
    :data:`RELATION_KIND_RE` and may not be ``mention``; a bad kind or a
    non-string target is reported in ``problems`` and skipped, never
    guessed. Order is the frontmatter's, duplicates dropped.
    """
    pairs: List[Tuple[str, str]] = []
    problems: List[str] = []
    if value is None:
        return pairs, problems
    if not isinstance(value, Mapping):
        return pairs, [f"relations must be a mapping of kind -> targets, got {type(value).__name__}"]
    seen: Set[Tuple[str, str]] = set()
    for kind, targets in value.items():
        kind = str(kind)
        if kind == MENTION or not RELATION_KIND_RE.match(kind):
            problems.append(f"relation kind {kind!r} is not allowed")
            continue
        items = targets if isinstance(targets, (list, tuple)) else [targets]
        for t in items:
            if t is None:
                continue
            if not isinstance(t, str) or not t.strip():
                problems.append(f"relation {kind}: target {t!r} is not a string")
                continue
            key = (kind, t.strip())
            if key not in seen:
                seen.add(key)
                pairs.append(key)
    return pairs, problems


def resolve_relation_target(
    raw: str, source_id: str, maps, known_ids: Set[str], candidates,
    path_ids: Optional[Mapping[str, str]] = None,
) -> Tuple[Optional[str], Optional[str], List[str], str]:
    """Resolve one ``relations:`` target: an id, a path, or a wiki name.

    Returns ``(target, reason, candidate_ids, recorded_target)``. Exact ids
    win (``finding/01J9``), then path readings (``../x.md``), then the name
    index — so a relation can name a concept the way a person would.
    """
    ref = raw.strip()
    if ref.startswith("[[") and ref.endswith("]]"):
        ref = ref[2:-2].split("|", 1)[0].strip()
    if ref in known_ids:
        return ref, None, [], ref
    if ref.lower().endswith(".md"):
        hit, recorded = resolve_path_link(ref, source_id, known_ids, path_ids)
        if hit is not None:
            return hit, None, [], hit
        return None, "missing", [], recorded
    hit, reason, cands = resolve_wiki_detail(ref, maps, known_ids, candidates)
    return hit, reason, cands, (hit or ref)
