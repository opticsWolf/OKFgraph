"""okf-asset:// staging convention for extracted images.

Rewrites ``![](local_path)`` links to ``![](okf-asset://<id>)`` and copies
bytes into ``<out_dir>/_assets/<id>.<ext>``. Does no embedding or database
work — that is okfgraph's job at ingest time.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path
from typing import Tuple

ASSET_STORE_DIRNAME = "_assets"

_MD_IMAGE_RE = re.compile(r"!\[(?P<alt>.*?)\]\((?P<src>.*?)\)")


def _split_src_and_title(raw: str) -> str:
    """Return just the path/URL part of a markdown image target."""
    raw = raw.strip()
    if not raw:
        return raw
    m = re.match(r"^(<[^>]+>|\S+)", raw)
    part = m.group(1) if m else raw
    if part.startswith("<") and part.endswith(">"):
        part = part[1:-1]
    return part


def asset_id(concept_stem: str, occurrence: int, img_bytes: bytes) -> str:
    """Deterministic, concept-scoped asset id.

    Scoping by the owning document keeps ids unique per concept; hashing the
    bytes keeps re-runs on unchanged input idempotent.
    """
    h = hashlib.sha256()
    h.update(f"{concept_stem}|{occurrence}|".encode("utf-8"))
    h.update(img_bytes)
    return f"img_{h.hexdigest()[:16]}"


def stage_images_as_okf_assets(
    md: str,
    image_src_dir: Path,
    source_path: Path,
    out_dir: Path,
    concept_stem: str,
) -> Tuple[str, int]:
    """Move extracted images into ``<out_dir>/_assets/<id>.<ext>`` and rewrite
    their links to ``okf-asset://<id>``.

    Returns ``(rewritten_md, count)``.
    """
    assets_dir = out_dir / ASSET_STORE_DIRNAME
    occurrence = {"n": 0}

    def _repl(match: "re.Match") -> str:
        alt = match.group("alt").strip()
        src = _split_src_and_title(match.group("src"))
        low = src.lower()
        if (
            not src
            or low.startswith(("http://", "https://", "okf-asset://", "data:"))
            or src.startswith(f"{ASSET_STORE_DIRNAME}/")
        ):
            return match.group(0)

        # Resolve the physical file: extractor temp dir, then doc-relative.
        cand = image_src_dir / Path(src).name
        if not cand.is_file():
            cand = source_path.parent / src
        if not cand.is_file() and Path(src).is_absolute():
            cand = Path(src)
        if not cand.is_file():
            return match.group(0)  # unresolved — leave the link untouched

        occurrence["n"] += 1
        data = cand.read_bytes()
        aid = asset_id(concept_stem, occurrence["n"], data)
        suffix = cand.suffix.lower() or ".bin"
        assets_dir.mkdir(parents=True, exist_ok=True)
        dest = assets_dir / f"{aid}{suffix}"
        if not dest.exists():
            shutil.copy2(cand, dest)
        return f"![{alt}](okf-asset://{aid})"

    new_md = _MD_IMAGE_RE.sub(_repl, md)
    return new_md, occurrence["n"]
