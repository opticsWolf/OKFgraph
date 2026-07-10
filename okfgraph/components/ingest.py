from __future__ import annotations

import base64
import hashlib
import heapq
import json
import logging
import math
import os
import re
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Set
from urllib.parse import urlparse

import mordant
import numpy as np
import yaml
import frontmatter
from okfgraph.models import ChunkModel, ConceptModel

logger = logging.getLogger(__name__)

class IngestManager:
    def __init__(self, _write_lock_ctx, bundle_root, device, import_mgr, delta_mgr):
        self._write_lock_ctx = _write_lock_ctx
        self.bundle_root = bundle_root
        self.device = device
        self.import_mgr = import_mgr
        self.delta_mgr = delta_mgr

    def _ingest_md_inner(
        self,
        md_path: str | Path,
        concept_id: str | None,
        title: str | None,
        description: str | None,
        tags: list[str] | None,
        mode: str,
    ) -> Dict[str, Any]:
        """Inner implementation of ingest_md (called under write lock)."""
        md_path = Path(md_path)
        if not md_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {md_path}")

        # Lint the file (may auto-fix in-place)
        lint_result = self._lint_converted_md(md_path, auto_fix=True)
        if lint_result["fixed"]:
            md_path.write_text(lint_result["content"], encoding="utf-8")

        # Parse frontmatter
        post = frontmatter.load(md_path)
        fm = dict(post.metadata)

        # Determine metadata
        cid = concept_id or md_path.stem.replace(" ", "_").lower()
        t = title or fm.get("title") or md_path.stem
        desc = description or fm.get("description") or fm.get("summary") or ""
        file_tags = fm.get("tags", [])
        all_tags = list(set((tags or []) + file_tags))

        # Build ConceptModel
        concept = ConceptModel.model_validate({
            "id": cid,
            "title": t,
            "description": desc,
            "body": post.content,
            "type": fm.get("type", "note"),
            "tags": all_tags,
        })

        # Import via shared single-concept pipeline
        result = self.import_mgr._import_single_concept(concept, post.content, mode)

        return {
            "concept_id": result["concept_id"],
            "title": result["title"],
            "description": result["description"],
            "tags": result["tags"],
            "chunk_count": result["chunk_count"],
            "image_count": result["image_count"],
            "lint_issues": {
                "fixed_count": lint_result["fixed_count"],
                "unfixable_count": len(lint_result["unfixable"]),
                "error_count": len(lint_result["errors"]),
            },
        }


    def _ingest_thoughts_inner(
        self,
        thoughts: str,
        topic: str,
        concept_id: str | None,
        tags: list[str] | None,
    ) -> Dict[str, Any]:
        """Inner implementation of ingest_thoughts (called under write lock)."""
        import uuid

        # Generate concept_id from topic if not provided
        if not concept_id:
            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            slug = topic.lower().replace(" ", "_")[:30]
            concept_id = f"thought_{slug}_{ts}_{str(uuid.uuid4())[:6]}"

        # Build OKF-compliant markdown
        header_lines = [
            "---",
            f'title: "Thought: {topic}"',
            "type: thought",
            "thought_type: reasoning",
            f"topic: {topic}",
            f"tags: [thought, reasoning, {topic}]",
            f"created: {datetime.now().isoformat()}",
            "---",
            "",
            thoughts,
        ]
        markdown = "\n".join(header_lines)

        # Defensive lint: the LLM-provided thoughts text may contain
        # malformed markdown (trailing spaces, blank lines, etc.).
        lint_result = self._lint_converted_md_str(markdown, auto_fix=True)
        if lint_result["fixed"]:
            markdown = lint_result["content"]
            logger.info(
                "ingest_thoughts: linted %s, fixed %d issues",
                concept_id,
                lint_result["fixed_count"],
            )
        if lint_result["errors"]:
            logger.warning(
                "ingest_thoughts: %s has %d structural errors",
                concept_id,
                len(lint_result["errors"]),
            )

        # Apply tags
        all_tags = list(set(["thought", "reasoning", topic] + (tags or [])))

        # Build ConceptModel
        concept = ConceptModel.model_validate({
            "id": concept_id,
            "title": f"Thought: {topic}",
            "description": f"Reasoning about {topic}",
            "body": markdown,
            "type": "thought",
            "tags": all_tags,
        })

        # Import via shared single-concept pipeline
        result = self.import_mgr._import_single_concept(concept, markdown, "text")

        return {
            "concept_id": result["concept_id"],
            "topic": topic,
            "tags": result["tags"],
            "chunk_count": result["chunk_count"],
            "markdown": markdown,
            "lint_issues": lint_result,
        }

    def _lint_converted_md(
        self,
        md_path: Path,
        *,
        auto_fix: bool = True,
    ) -> Dict[str, Any]:
        """Lint a markdown file and optionally auto-fix fixable issues.

        Returns a dict with:
        - "content": the (possibly fixed) markdown content (str)
        - "fixed": whether content was modified (bool)
        - "fixed_count": number of auto-fixed issues (int)
        - "unfixable": list of unfixable diagnostics (list)
        - "errors": list of error-level diagnostics (list)
        """
        content = md_path.read_text(encoding="utf-8")
        diagnostics = mordant.lint(content, gfm_opts=mordant.GfmOptions.all())

        if not diagnostics:
            return {
                "content": content,
                "fixed": False,
                "fixed_count": 0,
                "unfixable": [],
                "errors": [],
            }

        # Categorize diagnostics
        fixable_rules = {"MD009", "MD012", "MD047"}  # whitespace/formatting
        error_rules = {"MD001", "MD031", "MD033"}    # structural errors
        unfixable = [d for d in diagnostics if d.rule not in fixable_rules]
        errors = [d for d in diagnostics if d.rule in error_rules]

        fixed_content = content
        fixed_count = 0

        if auto_fix:
            fixable = [d for d in diagnostics if d.rule in fixable_rules]
            if fixable:
                result = mordant.fix(content, gfm_opts=mordant.GfmOptions.all())
                if result.fixed:
                    fixed_content = result.output
                    fixed_count = len(result.fixed)
                    logger.info(
                        "auto-fixed %d issues in %s",
                        fixed_count,
                        md_path.name,
                    )

        if unfixable:
            logger.warning(
                "%d unfixable issues in %s: %s",
                len(unfixable),
                md_path.name,
                ", ".join(f"{d.rule} (line {d.line})" for d in unfixable[:5]),
            )

        if errors:
            logger.warning(
                "%d structural errors in %s — import may produce unexpected results: %s",
                len(errors),
                md_path.name,
                ", ".join(f"{d.rule} (line {d.line})" for d in errors[:3]),
            )

        return {
            "content": fixed_content,
            "fixed": fixed_count > 0,
            "fixed_count": fixed_count,
            "unfixable": unfixable,
            "errors": errors,
        }


    def _lint_converted_md_str(
        self,
        content: str,
        *,
        auto_fix: bool = True,
    ) -> Dict[str, Any]:
        """Lint markdown content in-memory (no file I/O).

        Returns a dict with the same keys as ``_lint_converted_md``.
        """
        diagnostics = mordant.lint(content, gfm_opts=mordant.GfmOptions.all())

        if not diagnostics:
            return {
                "content": content,
                "fixed": False,
                "fixed_count": 0,
                "unfixable": [],
                "errors": [],
            }

        fixable_rules = {"MD009", "MD012", "MD047"}
        error_rules = {"MD001", "MD031", "MD033"}
        unfixable = [d for d in diagnostics if d.rule not in fixable_rules]
        errors = [d for d in diagnostics if d.rule in error_rules]

        fixed_content = content
        fixed_count = 0

        if auto_fix:
            fixable = [d for d in diagnostics if d.rule in fixable_rules]
            if fixable:
                result = mordant.fix(content, gfm_opts=mordant.GfmOptions.all())
                if result.fixed:
                    fixed_content = result.output
                    fixed_count = len(result.fixed)

        return {
            "content": fixed_content,
            "fixed": fixed_count > 0,
            "fixed_count": fixed_count,
            "unfixable": [d.rule for d in unfixable],
            "errors": [d.rule for d in errors],
        }


    def ingest_md(
        self,
        md_path: str | Path,
        *,
        concept_id: str | None = None,
        title: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        mode: str = "text",
    ) -> Dict[str, Any]:
        """Import a single markdown file into the knowledge graph.

        This is the programmatic counterpart to ``import_bundle()`` but
        operates on a single file with explicit metadata control.

        The file is linted with mordant before import. Fixable issues
        (MD009, MD012, MD047) are auto-corrected. Unfixable issues
        are logged as warnings but do not block import.

        Args:
            md_path: Path to the markdown file to import.
            concept_id: Optional explicit concept ID. If None, generated from filename.
            title: Optional title override (defaults to frontmatter or filename).
            description: Optional description override (defaults to frontmatter).
            tags: Optional tags to apply to the concept.
            mode: Image ingestion mode (text | optional | omni).

        Returns:
            Dict with keys:
            - "concept_id": The imported concept ID
            - "title": Title used
            - "description": Description used
            - "tags": Applied tags
            - "chunk_count": Number of chunks created (if chunking enabled)
            - "image_count": Number of images ingested
            - "lint_issues": Lint result dict (fixed_count, unfixable, errors)
        """
        # Acquire write lock (Gap #7b)
        with self._write_lock_ctx():
            return self._ingest_md_inner(md_path, concept_id, title, description, tags, mode)


    def ingest_pdf(
        self,
        pdf_path: str | Path,
        *,
        auto_import: bool = True,
        output_dir: str | Path | None = None,
        routing_mode: str = "auto",
        mode: str = "text",
        batch_size: int = 32,
        purge_deleted: bool = False,
        extract_images: bool = True,
        on_page: Callable[[int, int], None] | None = None,
    ) -> Dict[str, Any]:
        """Convert a PDF to markdown and optionally import into the graph.

        This is the programmatic counterpart to the ``okf ingest`` CLI command.
        It uses the HybridConverter pipeline (pdf_oxide fast path + ONNX/Rapid
        heavy passes) to convert the PDF, then optionally imports the resulting
        markdown into the knowledge graph via ``import_bundle()``.

        Args:
            pdf_path: Path to the PDF file.
            auto_import: If True, import the converted markdown into the graph.
                If False, write to disk only.
            output_dir: Output directory for the markdown (used when
                auto_import=False). Defaults to the PDF's parent directory.
            routing_mode: ONNX routing mode — "auto", "surgical", "always",
                or "never". Controls when ONNX models are invoked.
            mode: Image ingestion mode for auto-import — "text", "optional",
                or "omni". Only used when auto_import=True.
            batch_size: Batch size for encoding during auto-import.
            purge_deleted: If True, purge deleted concepts during auto-import.
            extract_images: Whether to extract embedded images from the PDF.
            on_page: Optional callback(page_index, page_total) for progress.

        Returns:
            A dict with keys:
            - "md_path": Path to the converted markdown file (always present)
            - "concept_ids": List of imported concept IDs (only when auto_import=True)
            - "image_dir": Path to the staged images directory (always present)
            - "page_count": Number of pages in the PDF

        Raises:
            RuntimeError: If pdf_oxide is not installed.
        """
        from tempfile import TemporaryDirectory

        from okfgraph.ingest import ConverterConfig, HybridConverter, RoutingMode
        from okfgraph.ingest.assets import stage_images_as_okf_assets

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Build converter config
        config = ConverterConfig(
            routing_mode=RoutingMode(routing_mode),
            device=self.device,
            extract_images=extract_images,
        )

        converter = HybridConverter(config, log=logger.info)

        try:
            # Ensure ONNX models are loaded (if routing mode requires them)
            converter.ensure_models()

            if auto_import:
                # Auto-import: convert to temp dir, import into graph, clean up.
                with TemporaryDirectory(prefix="okf_ingest_") as tmp:
                    work_dir = Path(tmp)
                    logger.info("converting %s → %s", pdf_path, work_dir)

                    md = converter.convert_pdf(
                        path=pdf_path,
                        work_dir=work_dir,
                        should_continue=lambda: True,
                        on_page=on_page or (lambda idx, total: logger.info(
                            "page %d/%d", idx + 1, total
                        )),
                    )

                    # Write markdown to temp dir
                    stem = pdf_path.stem
                    md_path = work_dir / f"{stem}.md"
                    md_path.write_text(md, encoding="utf-8")

                    # Stage any extracted images as okf-asset:// URIs
                    img_dir = work_dir / "_assets"
                    img_dir.mkdir(exist_ok=True)
                    for img in work_dir.glob("*"):
                        if img.is_file() and img.suffix.lower() in (
                            ".png", ".jpg", ".jpeg", ".gif", ".webp",
                        ):
                            img.rename(img_dir / img.name)

                    md_text = md_path.read_text(encoding="utf-8")
                    stage_images_as_okf_assets(
                        md_text, work_dir, pdf_path, work_dir, stem
                    )

                    # Lint converted markdown (Gap #5c)
                    lint_result = self._lint_converted_md(md_path, auto_fix=True)
                    if lint_result["fixed"]:
                        md_path.write_text(lint_result["content"], encoding="utf-8")
                        logger.info(
                            "linted %s: fixed %d issues",
                            md_path.name,
                            lint_result["fixed_count"],
                        )
                    if lint_result["errors"]:
                        logger.warning(
                            "PDF output has %d structural errors — proceeding anyway",
                            len(lint_result["errors"]),
                        )

                    # Import into the graph
                    # Temporarily override bundle_root for the temp directory
                    old_bundle_root = self.bundle_root
                    self.bundle_root = work_dir
                    # Keep the injected DeltaDetector and ImportManager in sync:
                    # each stores its own bundle_root copy, and import_bundle /
                    # _changed_directories rely on it (Phase 3 refactor).
                    self.delta_mgr.bundle_root = work_dir
                    self.import_mgr.bundle_root = work_dir
                    try:
                        ids = self.import_mgr.import_bundle(
                            work_dir,
                            batch_size=batch_size,
                            mode=mode,
                            purge_deleted=purge_deleted,
                        )
                    finally:
                        self.bundle_root = old_bundle_root
                        self.delta_mgr.bundle_root = old_bundle_root
                        self.import_mgr.bundle_root = old_bundle_root
                    logger.info("imported %d concept(s) from %s", len(ids), pdf_path)

                    return {
                        "md_path": str(md_path),
                        "concept_ids": ids,
                        "image_dir": str(img_dir),
                        "page_count": len(list(Path(tmp).rglob("*.md"))),
                    }
            else:
                # Output-only: convert to the specified output directory.
                output_dir = Path(output_dir) if output_dir else pdf_path.parent
                output_dir.mkdir(parents=True, exist_ok=True)

                logger.info("converting %s → %s", pdf_path, output_dir)

                md = converter.convert_pdf(
                    path=pdf_path,
                    work_dir=output_dir,
                    should_continue=lambda: True,
                    on_page=on_page or (lambda idx, total: logger.info(
                        "page %d/%d", idx + 1, total
                    )),
                )

                # Write markdown
                stem = pdf_path.stem
                md_path = output_dir / f"{stem}.md"
                md_path.write_text(md, encoding="utf-8")

                # Stage images
                img_dir = output_dir / "_assets"
                img_dir.mkdir(exist_ok=True)
                for img in output_dir.glob("*"):
                    if img.is_file() and img.suffix.lower() in (
                        ".png", ".jpg", ".jpeg", ".gif", ".webp",
                    ):
                        img.rename(img_dir / img.name)

                md_text = md_path.read_text(encoding="utf-8")
                stage_images_as_okf_assets(
                    md_text, output_dir, pdf_path, output_dir, stem
                )

                logger.info("written %s", md_path)
                logger.info("assets in %s", img_dir)

                return {
                    "md_path": str(md_path),
                    "concept_ids": [],
                    "image_dir": str(img_dir),
                    "page_count": len(list(output_dir.rglob("*.md"))),
                }
        finally:
            converter.close()


    def ingest_thoughts(
        self,
        thoughts: str,
        *,
        topic: str,
        concept_id: str | None = None,
        tags: list[str] | None = None,
    ) -> Dict[str, Any]:
        """Store LLM reasoning/thinking as a searchable concept.

        Wraps the raw reasoning text in OKF-compliant markdown with metadata
        (type=thought, thought_type=reasoning, topic) so it can be searched,
        traversed, and used as context for other queries.

        The markdown is linted with mordant before import. Fixable issues
        are auto-corrected.

        Args:
            thoughts: The raw reasoning text from the LLM.
            topic: High-level topic or domain for the reasoning.
            concept_id: Optional explicit concept ID. If None, generated from topic.
            tags: Optional additional tags.

        Returns:
            Dict with keys:
            - "concept_id": The created concept ID
            - "topic": Topic used
            - "tags": Applied tags
            - "chunk_count": Number of chunks created
            - "markdown": The generated markdown content
        """
        # Acquire write lock (Gap #7b)
        with self._write_lock_ctx():
            return self._ingest_thoughts_inner(thoughts, topic, concept_id, tags)

