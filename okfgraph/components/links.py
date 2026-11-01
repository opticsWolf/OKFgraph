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
from urllib.parse import unquote

#: Source files a concept can come from; link targets must end in one of these.
SOURCE_EXTS = (".md", ".markdown", ".txt")

# ``[text](target)``, ``[text](<target with spaces>)`` and ``[text](t "title")``.
# A bare target cannot contain whitespace or ')', so one match never spans two
# links. Group 1 is the angle-bracket form, group 2 the bare form.
_MD_LINK_RE = re.compile(
    r"\[[^\]]*\]\(\s*(?:<([^<>\n]+)>|([^)\s<>]+))(?:\s+\"[^\"]*\")?\s*\)"
)
WIKI_RE = re.compile(r"\[\[(.*?)\]\]")
SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
# Fenced blocks: an opener of 3+ backticks or tildes, closed by the same fence
# (or the end of the text). Single-line code spans: a backtick run closed by a
# run of exactly the same length.
_FENCE_RE = re.compile(
    r"^( {0,3})(`{3,}|~{3,})[^\n]*\n.*?(?:^ {0,3}\2[`~]*[ \t]*$|\Z)",
    re.M | re.S,
)
_CODE_SPAN_RE = re.compile(r"(?<!`)(`+)(?!`)[^\n]+?(?<!`)\1(?!`)")


def _blank(match: "re.Match[str]") -> str:
    """Same-length run of spaces (newlines kept), so offsets stay valid."""
    return re.sub(r"[^\n]", " ", match.group(0))


def strip_code(body: str) -> str:
    """``body`` with fenced blocks and inline code spans blanked out."""
    return _CODE_SPAN_RE.sub(_blank, _FENCE_RE.sub(_blank, body))


def _path_part(raw: str) -> str:
    """The target without its ``#fragment`` or ``?query``."""
    return re.split(r"[#?]", raw, maxsplit=1)[0].strip()


def extract_md_links(body: str) -> List[str]:
    """Raw targets of ``[text](target)`` links that name a source file.

    External URLs are included (callers skip them with :func:`is_external`).
    Code blocks and code spans are never read as links.
    """
    out: List[str] = []
    for m in _MD_LINK_RE.finditer(strip_code(body or "")):
        raw = m.group(1) or m.group(2)
        if _path_part(raw).lower().endswith(SOURCE_EXTS):
            out.append(raw)
    return out


def extract_wikilinks(body: str) -> List[str]:
    """Raw targets of ``[[target]]`` / ``[[target|display]]`` (display stripped)."""
    return [
        m.split("|", 1)[0].strip()
        for m in WIKI_RE.findall(strip_code(body or ""))
    ]


def is_external(raw: str) -> bool:
    """True for ``scheme:...`` targets (http, mailto, okf-asset, ...)."""
    return bool(SCHEME_RE.match((raw.split("#", 1)[0]).strip()))


def _legacy_target(raw: str) -> str:
    """The pre-0.10.3 reading: the target with a substring ``.md`` removed."""
    return raw.lstrip("./").replace("\\", "/").replace(".md", "").lstrip("/")


def _candidate_id(raw: str, source_id: str) -> Optional[str]:
    """Concept id a link points to, judged from the source file's location.

    Decode the path, resolve it against the source's directory (a leading
    ``/`` means bundle-root-relative), normalise, and drop one source
    extension. ``None`` when the link leaves the bundle or names nothing.
    """
    path = unquote(_path_part(raw)).replace("\\", "/")
    if not path:
        return None
    if path.startswith("/"):
        joined = path.lstrip("/")
    else:
        joined = posixpath.join(posixpath.dirname(source_id), path)
    norm = posixpath.normpath(joined) if joined else "."
    if norm == ".":
        return None
    for ext in SOURCE_EXTS:
        if norm.lower().endswith(ext):
            norm = norm[: -len(ext)]
            break
    return norm


def link_target(
    raw: str, source_id: str, known_ids: Set[str],
    path_ids: Optional[Mapping[str, str]] = None,
) -> Tuple[Optional[str], str]:
    """Resolve one markdown link from ``source_id``.

    Returns ``(concept_id_or_None, target_to_record)``. The first is the id to
    link to; ``None`` means broken, and the second is the target to store on
    the broken-link row. The source-relative reading wins. A link that leaves
    the bundle is broken and never falls back. Otherwise, if the source-relative
    reading is not a known id, the pre-0.10.3 reading is used when that one is
    known, so bundles that relied on the root-relative reading keep resolving.

    ``path_ids`` maps a file path (without extension) to the concept id of
    the file there, for bundles where the two differ (frontmatter identity,
    0.11.0 O3): a Markdown link names a *file*, so it resolves to that
    file's concept.
    """
    cand = _candidate_id(raw, source_id)
    if cand is not None and (cand == ".." or cand.startswith("../")):
        return None, cand
    legacy = _legacy_target(_path_part(raw))
    for reading in (cand, legacy):
        if not reading:
            continue
        if reading in known_ids:
            return reading, reading
        if path_ids and reading in path_ids:
            return path_ids[reading], path_ids[reading]
    return None, cand if cand is not None else legacy


def normalize_path_link(raw: str) -> str:
    """Deprecated: use :func:`link_target`. Kept for existing imports.

    Gives the pre-0.10.3 reading with no source file and no known-id check.
    """
    return _legacy_target(_path_part(raw))


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
    if _path_part(ref).lower().endswith(SOURCE_EXTS):
        hit, recorded = link_target(ref, source_id, known_ids, path_ids)
        if hit is not None:
            return hit, None, [], hit
        return None, "missing", [], recorded
    hit, reason, cands = resolve_wiki_detail(ref, maps, known_ids, candidates)
    return hit, reason, cands, (hit or ref)
