"""Pre-import bundle gate: frontmatter + link validation, no DB, no model.

``lint_bundle()`` checks the file side (what ``import --all`` is *about*
to ingest); doctor checks the row side (what is *already* indexed).
Same ``links.py`` pure helpers, same resolution rules as import —
a lint-clean bundle must import with zero ``broken_link`` findings
(locked by the consistency test in ``tests/test_lint.py``).

Report shape (JSON-stable, all lists sorted)::
    {"dir": str, "files": int,
     "errors": [{"file", "rule", "message", ...}],
     "warnings": [{"file", "rule", "message"}],
     "skipped_hidden": int,  # 0.10.1 B2: concept files hidden by the
     "clean": bool}  # shared dot/tool-dir skip rule (diagnostic only)
     # True when errors is empty (warnings allowed)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import frontmatter

from okfgraph.components.import_ import (
    count_skipped_concept_files,
    in_skipped_dir,
    is_concept_file,
    parse_source_file,
)
from okfgraph.components.links import (
    build_name_index_full,
    extract_md_links,
    extract_wikilinks,
    frontmatter_relations,
    is_external,
    link_target,
    resolve_relation_target,
    resolve_wiki_detail,
)

logger = logging.getLogger(__name__)


def _err(file: str, rule: str, message: str, **extra: Any) -> Dict[str, Any]:
    item: Dict[str, Any] = {"file": file, "rule": rule, "message": message}
    item.update(extra)
    return item


def lint_bundle(bundle_dir: str | Path, identity: str = "path") -> Dict[str, Any]:
    """Validate a bundle directory without touching any database or model.

    0.11.0: ``identity`` matches import (O3) — under ``frontmatter`` two
    files with one ``id:`` are a ``duplicate_id`` error (import refuses the
    second). A duplicate ``id:`` is reported under ``path`` identity too:
    it makes the name ambiguous for ``[[wikilinks]]``. ``relations:`` (O6)
    are checked like links (``bad_relation``, ``dangling_relation``), and
    ambiguous names say which concepts claim them (O12).
    """
    root = Path(bundle_dir)
    files = sorted(fp for fp in root.rglob("*")
                    if is_concept_file(fp) and not in_skipped_dir(fp))
    # B2: count what the shared skip rule hid so a fully-filtered walk can
    # say so instead of reporting 0 files as clean silence.
    skipped_hidden = count_skipped_concept_files(root)

    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    parsed: List[Tuple[str, Any, str, str]] = []  # (cid, concept, body, rel)

    for fp in files:
        rel = str(fp.relative_to(root)).replace("\\", "/")
        try:
            concept, body, cid = parse_source_file(fp, root, identity=identity)
        except Exception as e:  # noqa: BLE001 — every failure mode is a finding
            errors.append(_err(rel, "parse", f"{type(e).__name__}: {e}"))
            continue
        try:
            raw_fm = dict(frontmatter.load(fp).metadata)
        except Exception as e:  # noqa: BLE001 — same finding class as above
            errors.append(_err(rel, "parse", f"{type(e).__name__}: {e}"))
            continue
        # Import synthesizes both (note / stem), so these warn — erroring
        # would contradict import behaviour (deliberate deviation from
        # google-okf, whose pipeline has no synthesis step).
        if not raw_fm.get("type"):
            warnings.append(_err(
                rel, "missing_type", "no 'type:' frontmatter; will import as 'note'"))
        if not raw_fm.get("title"):
            warnings.append(_err(
                rel, "missing_title", "no 'title:' frontmatter; will use filename stem"))
        parsed.append((cid, concept, body, rel))

    # One id, one file (O3). Under path identity the ids cannot collide, but
    # two files sharing a frontmatter id: still make that name ambiguous.
    seen_ids: Dict[str, str] = {}
    for cid, concept, _body, rel in parsed:
        extra = concept.model_extra or {}
        key = cid if identity == "frontmatter" else extra.get("uid")
        if not key:
            continue
        if key in seen_ids:
            errors.append(_err(
                rel, "duplicate_id",
                f"frontmatter id '{key}' is already used by {seen_ids[key]}"
                + ("; import refuses this file" if identity == "frontmatter"
                   else "; [[wikilinks]] to it are ambiguous"),
                id=key, other=seen_ids[key]))
        else:
            seen_ids[key] = rel

    # Resolution context mirrors ImportManager: known ids + name index.
    known_ids = {cid for cid, _, _, _ in parsed}
    path_ids = {}
    for cid, _concept, _body, rel in parsed:
        stem = rel.rsplit(".", 1)[0]
        if stem != cid:
            path_ids[stem] = cid
    index_entries = []
    for cid, concept, _, _ in parsed:
        extra = concept.model_extra or {}
        index_entries.append({
            "id": cid,
            "title": concept.title,
            "uid": extra.get("uid"),
            "aliases": extra.get("aliases", extra.get("alias")),
        })
    maps, _ambiguous, candidates = build_name_index_full(index_entries)

    for cid, concept, body, rel in parsed:
        # Same extraction + resolution as import: only *.md-anchored targets
        # match (source-relative, then root-relative); externals never
        # resolve-or-fail.
        for raw in extract_md_links(body):
            if is_external(raw):
                continue
            target, recorded = link_target(raw, cid, known_ids, path_ids)
            if target is None:
                errors.append(_err(
                    rel, "dangling_link",
                    "would import as BrokenLink (target has no concept)",
                    link=raw, target=recorded or "(empty)"))
        for raw in extract_wikilinks(body):
            if not raw or is_external(raw):
                continue
            hit, reason, cands = resolve_wiki_detail(raw, maps, known_ids, candidates)
            if hit is None:
                errors.append(_err(
                    rel, "dangling_wikilink",
                    ("would import as BrokenLink (name is ambiguous: "
                     f"{', '.join(cands)})") if reason == "ambiguous" else
                    "would import as BrokenLink (name resolves to nothing)",
                    link=raw, reason=reason, candidates=cands))
        pairs, problems = frontmatter_relations(
            (concept.model_extra or {}).get("relations"))
        for problem in problems:
            errors.append(_err(rel, "bad_relation", problem))
        for kind, raw in pairs:
            hit, reason, cands, recorded = resolve_relation_target(
                raw, cid, maps, known_ids, candidates, path_ids)
            if hit is None:
                errors.append(_err(
                    rel, "dangling_relation",
                    f"relation {kind} would import as BrokenLink ({reason})",
                    link=raw, kind=kind, reason=reason, candidates=cands,
                    target=recorded))

    errors.sort(key=lambda e: (e["file"], e["rule"], e["message"]))
    warnings.sort(key=lambda w: (w["file"], w["rule"], w["message"]))
    return {
        "dir": str(root),
        "files": len(files),
        "errors": errors,
        "warnings": warnings,
        "skipped_hidden": skipped_hidden,
        "clean": not errors,
    }
