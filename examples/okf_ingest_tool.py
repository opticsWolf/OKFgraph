"""Convert collected documents into OKF Markdown and hand their images to the
OKF graph via the ``okf-asset://`` protocol.

Division of labour (important):

* **This tool** does the one thing the graph layer cannot: it cracks open
  binary documents (PDF / Office) and extracts their embedded image *bytes*.
  Each extracted image is written to the bundle's on-disk asset store
  (``<bundle>/_assets/<asset_id>.<ext>``) and its markdown link is rewritten to
  ``![alt](okf-asset://<asset_id>)``.

* **okfgraph (OKFRouter)** owns everything else about images: it resolves each
  ``okf-asset://`` reference back to those bytes, embeds them in the unified
  jina-embeddings-v5 vector space according to the selected ingestion *mode*
  (``text`` / ``optional`` / ``omni``), stores the BLOB, builds the
  ``INCLUDES_ASSET`` edges, dedupes by content hash, and prunes removed assets.

The tool therefore does **no** embedding, no ``ImageAsset`` writes, no edge
creation, and no schema management — that would duplicate work the router
already performs (and, historically, with a different/incompatible model,
dimension, and index).
"""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import tempfile
import yaml
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from tool_box import ToolBox
    from tools.file_sandbox import FileToolSandbox

# Optional dependencies for rich documents
try:
    from pdf_oxide import PdfDocument
except ImportError:
    PdfDocument = None

try:
    from office_oxide import Document as OfficeDocument
except ImportError:
    OfficeDocument = None

# okfgraph owns all image embedding / storage / linking. When it is importable
# we drive it in-process; otherwise we fall back to the ``okf`` CLI.
try:
    from okfgraph.router import OKFRouter
    from okfgraph.images import ASSET_STORE_DIRNAME
    HAS_OKFGRAPH = True
except ImportError:
    OKFRouter = None
    ASSET_STORE_DIRNAME = "_assets"
    HAS_OKFGRAPH = False


OFFICE_EXTS = {".docx", ".xlsx", ".pptx", ".doc", ".xls", ".ppt"}

# Loose image files are assets, not documents — skip them in the document walk
# (they are still resolved on demand when a converted doc references them).
IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp",
    ".tif", ".tiff", ".svg", ".avif", ".heic", ".heif",
}

# Maps the legacy ``clip_mode`` values onto the unified ingestion modes.
_LEGACY_CLIP_TO_MODE = {
    "never": "text",
    "on_missing_description": "optional",
    "always": "omni",
}
_VALID_MODES = {"text", "optional", "omni"}

# ``![alt](src)`` — src may carry an optional "title" we strip off.
_IMG_LINK_RE = re.compile(r'!\[(?P<alt>.*?)\]\((?P<src>.*?)\)')


# ── Helpers ─────────────────────────────────────────────────────────────

def _resolve_mode(image_mode: Optional[str], clip_mode: Optional[str]) -> str:
    """Resolve the effective ingestion mode.

    ``image_mode`` (text|optional|omni) takes precedence; otherwise the legacy
    ``clip_mode`` is mapped onto it for backward compatibility.
    """
    if image_mode:
        m = image_mode.strip().lower()
        if m not in _VALID_MODES:
            raise ValueError(
                f"Invalid image_mode '{image_mode}'. Choose one of: "
                f"{', '.join(sorted(_VALID_MODES))}."
            )
        return m
    return _LEGACY_CLIP_TO_MODE.get((clip_mode or "").strip().lower(), "optional")


def _asset_id(concept_stem: str, occurrence: int, img_bytes: bytes) -> str:
    """Deterministic, concept-scoped asset id.

    Scoping by the owning concept keeps ids unique per document (so the router's
    per-asset delete-then-create never clobbers an asset shared across concepts),
    while hashing the bytes keeps re-runs on unchanged input idempotent.
    """
    h = hashlib.sha256()
    h.update(f"{concept_stem}|{occurrence}|".encode("utf-8"))
    h.update(img_bytes)
    return f"img_{h.hexdigest()[:16]}"


def _split_src_and_title(raw: str) -> str:
    """Return just the path/URL part of a markdown link target."""
    raw = raw.strip()
    if not raw:
        return raw
    m = re.match(r'^(<[^>]+>|\S+)', raw)
    part = m.group(1) if m else raw
    if part.startswith("<") and part.endswith(">"):
        part = part[1:-1]
    return part


# ── Pydantic request schemas ────────────────────────────────────────────

class IngestOKFArgs(BaseModel):
    source_dir: str = Field(
        "collection",
        description="Directory containing raw files to ingest (relative to sandbox root)."
    )
    bundle_dir: str = Field(
        "okf_bundle",
        description="Destination directory where generated .md OKF concepts will be stored."
    )
    db_path: str = Field(
        "okfgraph.db",
        description="Path to the LadybugDB database file."
    )
    concept_type: str = Field(
        "document",
        description="Default concept type to assign in the OKF frontmatter."
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags to automatically apply to the ingested documents."
    )
    image_mode: Optional[str] = Field(
        None,
        description=(
            "How images are embedded into the unified vector index: "
            "'text' (alt-text / filename only, no multimodal model), "
            "'optional' (multimodal model only for images lacking alt-text), or "
            "'omni' (multimodal model for every image). "
            "Defaults to 'optional' (or the mapped legacy clip_mode)."
        ),
    )
    clip_mode: str = Field(
        "on_missing_description",
        description=(
            "[Deprecated — use image_mode] Legacy selector mapped onto image_mode: "
            "'never'->text, 'on_missing_description'->optional, 'always'->omni."
        ),
    )


# ── Pydantic response schemas ───────────────────────────────────────────

class IngestOKFResponse(BaseModel):
    """Structured reply from ingest_to_okf."""
    processed_count: int = Field(..., description="Number of source documents converted.")
    images_extracted: int = Field(..., description="Images pulled from documents into the bundle asset store.")
    clip_embeddings_generated: int = Field(..., description="Images embedded by the multimodal (omni) model. Back-compat alias of images_via_omni.")
    ingested_count: int | None = Field(..., description="Number of concepts successfully ingested into the OKF graph.")
    generated_md_files: list[str] = Field(default_factory=list, description="Relative paths of generated OKF files.")
    errors: list[str] = Field(default_factory=list, description="List of errors encountered.")
    image_mode: str | None = Field(None, description="Effective image ingestion mode used.")
    images_ingested: int | None = Field(None, description="Image assets stored by the router (None if okfgraph unavailable).")
    images_via_omni: int = Field(0, description="Images embedded via the omni multimodal model.")
    images_via_text: int = Field(0, description="Images embedded via the text model (alt-text / filename).")


# ── Tool implementation ─────────────────────────────────────────────────

def _ingest_to_okf_impl(
    sandbox: "FileToolSandbox", source_dir: str, bundle_dir: str,
    db_path: str, concept_type: str, tags: list[str],
    image_mode: Optional[str], clip_mode: str,
) -> IngestOKFResponse:

    def _err(msg: str) -> IngestOKFResponse:
        return IngestOKFResponse(
            processed_count=0, images_extracted=0, clip_embeddings_generated=0,
            ingested_count=0, generated_md_files=[], errors=[msg],
        )

    try:
        mode = _resolve_mode(image_mode, clip_mode)
    except ValueError as e:
        return _err(str(e))

    try:
        sandbox.check_read()
        sandbox.check_write()

        src_path = sandbox.resolve_path(source_dir)
        bundle_path = sandbox.resolve_path(bundle_dir)
        db_full_path = sandbox.resolve_path(db_path)
    except (ValueError, PermissionError) as e:
        return _err(str(e))

    if not src_path.is_dir():
        return _err(f"Source directory '{src_path}' does not exist.")

    assets_dir = bundle_path / ASSET_STORE_DIRNAME
    if not sandbox.dry_run:
        bundle_path.mkdir(parents=True, exist_ok=True)
        assets_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    images_extracted = 0
    generated_files: list[str] = []
    errors: list[str] = []

    # 1. Convert raw documents; extract image bytes into the bundle asset store.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_img_path = Path(tmp_dir)

        for file_path in sorted(src_path.rglob("*")):
            if not file_path.is_file():
                continue

            ext = file_path.suffix.lower()
            if ext in IMAGE_EXTS:
                continue  # an asset, not a document
            content = ""
            try:
                # ── Step A: extract text + images from the document
                if ext == ".pdf":
                    if PdfDocument is None:
                        errors.append(f"Skipped {file_path.name}: pdf_oxide not installed.")
                        continue
                    with PdfDocument(str(file_path)) as doc:
                        content = doc.to_markdown_all(
                            preserve_layout=False, detect_headings=True,
                            include_images=True, embed_images=False, image_out_dir=str(tmp_img_path),
                        )
                elif ext in OFFICE_EXTS:
                    if OfficeDocument is None:
                        errors.append(f"Skipped {file_path.name}: office_oxide not installed.")
                        continue
                    with OfficeDocument.open(str(file_path)) as doc:
                        # Not all office_oxide versions accept image extraction;
                        # fall back to a plain conversion if the kwarg is unknown.
                        try:
                            content = doc.to_markdown(image_out_dir=str(tmp_img_path))
                        except TypeError:
                            content = doc.to_markdown()
                else:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")

                # ── Step B: pick the destination concept first so asset ids are
                #            scoped to the final concept id (the .md stem).
                dest_file = bundle_path / f"{file_path.stem}.md"
                counter = 1
                while dest_file.exists():
                    dest_file = bundle_path / f"{file_path.stem}_v{counter}.md"
                    counter += 1
                concept_stem = dest_file.stem

                # ── Step C: stage each image into _assets/<id>.<ext> and rewrite
                #            its link to okf-asset://<id>. No embedding/DB work here.
                occurrence = {"n": 0}

                def _stage_image(match: "re.Match") -> str:
                    alt_text = match.group("alt").strip()
                    src = _split_src_and_title(match.group("src"))
                    low = src.lower()
                    if (not src
                            or low.startswith(("http://", "https://", "okf-asset://", "data:"))
                            or src.startswith(f"{ASSET_STORE_DIRNAME}/")):
                        return match.group(0)

                    # Resolve the physical file: extractor temp dir, then doc-relative.
                    cand = tmp_img_path / Path(src).name
                    if not cand.is_file():
                        cand = file_path.parent / src
                    if not cand.is_file() and Path(src).is_absolute():
                        cand = Path(src)
                    if not cand.is_file():
                        return match.group(0)  # unresolved — leave link untouched

                    occurrence["n"] += 1
                    nonlocal images_extracted
                    images_extracted += 1

                    if sandbox.dry_run:
                        return match.group(0)

                    img_bytes = cand.read_bytes()
                    asset_id = _asset_id(concept_stem, occurrence["n"], img_bytes)
                    suffix = cand.suffix.lower() or ".bin"
                    dest = assets_dir / f"{asset_id}{suffix}"
                    if not dest.exists():
                        shutil.copy2(cand, dest)
                    return f"![{alt_text}](okf-asset://{asset_id})"

                content = _IMG_LINK_RE.sub(_stage_image, content)

                # ── Step D: frontmatter + save
                if not sandbox.dry_run:
                    if not content.strip().startswith("---"):
                        fm_data = {
                            "title": file_path.stem,
                            "type": concept_type,
                            "tags": tags,
                            "resource": file_path.name,
                        }
                        fm_yaml = yaml.dump(fm_data, sort_keys=False, allow_unicode=True)
                        final_md = f"---\n{fm_yaml}---\n\n{content}"
                    else:
                        final_md = content
                    dest_file.write_text(final_md, encoding="utf-8")
                    generated_files.append(str(dest_file.relative_to(sandbox.root_dir)))
                else:
                    generated_files.append(f"[Dry-run] Would create OKF md: {dest_file.name}")

                processed += 1

            except Exception as e:
                errors.append(f"Error converting {file_path.name}: {e}")

    # 2. Hand the bundle to okfgraph — it does ALL image embedding/storage/linking.
    ingested_count: int | None = None
    images_ingested: int | None = None
    images_via_omni = 0
    images_via_text = 0

    if not sandbox.dry_run and processed > 0:
        try:
            if HAS_OKFGRAPH:
                # Checkpoint + close the DB when done so a subsequent reader
                # (e.g. the search browser, a separate process) can open the
                # file without a corrupted-WAL error. close() is called
                # defensively so this also works with lightweight router stand-
                # ins. import_bundle rebuilds the search indexes once per batch.
                router = OKFRouter(db_path=str(db_full_path), bundle_root=str(bundle_path))
                try:
                    imported_ids = router.import_bundle(bundle_path=bundle_path, mode=mode)
                    ingested_count = len(imported_ids)
                    images_ingested = 0
                    for cid in imported_ids:
                        for im in router.list_images(cid):
                            images_ingested += 1
                            route = (im.get("embed_route") or "").lower()
                            if route == "omni":
                                images_via_omni += 1
                            elif route == "text":
                                images_via_text += 1
                finally:
                    _close = getattr(router, "close", None)
                    if callable(_close):
                        try:
                            _close()
                        except Exception:
                            pass
            else:
                cmd = [
                    "okf", "import", "--all",
                    "--bundle", str(bundle_path),
                    "--db", str(db_full_path),
                    "--mode", mode,
                ]
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode == 0:
                    ingested_count = processed
                else:
                    errors.append(f"OKF CLI Error: {res.stderr.strip()}")
        except Exception as e:
            errors.append(f"Database Ingestion Error: {e}")

    return IngestOKFResponse(
        processed_count=processed,
        images_extracted=images_extracted,
        clip_embeddings_generated=images_via_omni,  # back-compat alias
        ingested_count=ingested_count,
        generated_md_files=generated_files,
        errors=errors,
        image_mode=mode,
        images_ingested=images_ingested,
        images_via_omni=images_via_omni,
        images_via_text=images_via_text,
    )


# ── Registration ────────────────────────────────────────────────────────

def attach_okf_ingest_tools(toolbox: "ToolBox", sandbox: "FileToolSandbox") -> None:
    """Mount the OKF ingestion tools onto a ToolBox instance."""
    toolbox._file_sandbox = sandbox  # type: ignore[attr-defined]

    procedure_base = sandbox.describe_policy()

    @toolbox.register(
        name="ingest_to_okfgraph",
        description="Convert a folder of raw documents into OKF Markdown and ingest them (with their images) into the unified Knowledge Graph.",
        args_model=IngestOKFArgs,
        procedure=(
            "Ingest raw documents into the unified OKF Knowledge Graph.\\n"
            "- Reads files from `source_dir` and converts them to OKF Markdown in `bundle_dir`.\\n"
            "- Extracts embedded images into the bundle asset store and references them via `okf-asset://` links.\\n"
            "- okfgraph then embeds each image into the unified vector index per `image_mode`,\\n"
            "  stores the bytes, and links concepts to assets — the tool does not embed or write assets itself.\\n"
            "- `image_mode`: text | optional | omni (legacy `clip_mode` is still accepted).\\n"
            f"\\n{procedure_base}"
        ),
    )
    def _ingest_to_okfgraph(
        db_pool: Any,
        user_session: dict,
        source_dir: str = "collection",
        bundle_dir: str = "okf_bundle",
        db_path: str = "okfgraph.db",
        concept_type: str = "document",
        tags: list[str] = None,
        image_mode: str = None,
        clip_mode: str = "on_missing_description",
    ) -> IngestOKFResponse:
        if tags is None:
            tags = []
        return _ingest_to_okf_impl(
            sandbox, source_dir, bundle_dir, db_path, concept_type, tags, image_mode, clip_mode,
        )
