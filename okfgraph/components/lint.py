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
     "clean": bool}  # True when errors is empty (warnings allowed)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import frontmatter

from okfgraph.components.import_ import (
    in_skipped_dir,
    is_concept_file,
    parse_source_file,
)
from okfgraph.components.links import (
    build_name_index,
    extract_md_links,
    extract_wikilinks,
    is_external,
    normalize_path_link,
    resolve_wiki,
)

logger = logging.getLogger(__name__)


def _err(file: str, rule: str, message: str, **extra: Any) -> Dict[str, Any]:
    item: Dict[str, Any] = {"file": file, "rule": rule, "message": message}
    item.update(extra)
    return item


def lint_bundle(bundle_dir: str | Path) -> Dict[str, Any]:
    """Validate a bundle directory without touching any database or model."""
    root = Path(bundle_dir)
    files = sorted(fp for fp in root.rglob("*")
                    if is_concept_file(fp) and not in_skipped_dir(fp))

    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    parsed: List[Tuple[str, Any, str, str]] = []  # (cid, concept, body, rel)

    for fp in files:
        rel = str(fp.relative_to(root)).replace("\\", "/")
        try:
            concept, body, cid = parse_source_file(fp, root)
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

    # Resolution context mirrors ImportManager: known ids + name index.
    known_ids = {cid for cid, _, _, _ in parsed}
    index_entries = []
    for cid, concept, _, _ in parsed:
        extra = concept.model_extra or {}
        index_entries.append({
            "id": cid,
            "title": concept.title,
            "uid": extra.get("uid"),
            "aliases": extra.get("aliases", extra.get("alias")),
        })
    maps, _ambiguous = build_name_index(index_entries)

    for cid, _concept, body, rel in parsed:
        # Same extraction + skip rules as _resolve_body_links: only
        # *.md-anchored targets match; externals never resolve-or-fail.
        for raw in extract_md_links(body):
            if is_external(raw):
                continue
            target = normalize_path_link(raw.split("#", 1)[0])
            if target not in known_ids:
                errors.append(_err(
                    rel, "dangling_link",
                    f"would import as BrokenLink (target has no concept)",
                    link=raw, target=target or "(empty)"))
        for raw in extract_wikilinks(body):
            if not raw or is_external(raw):
                continue
            if resolve_wiki(raw, maps, known_ids) is None:
                errors.append(_err(
                    rel, "dangling_wikilink",
                    "would import as BrokenLink (name resolves to nothing; "
                    "ambiguous names never resolve)",
                    link=raw))

    errors.sort(key=lambda e: (e["file"], e["rule"], e["message"]))
    warnings.sort(key=lambda w: (w["file"], w["rule"], w["message"]))
    return {
        "dir": str(root),
        "files": len(files),
        "errors": errors,
        "warnings": warnings,
        "clean": not errors,
    }
