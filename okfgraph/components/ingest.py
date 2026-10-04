from __future__ import annotations

import hashlib
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict

import mordant
import frontmatter
from okfgraph.models import ConceptModel, normalize_tags
from okfgraph.errors import OKFError

logger = logging.getLogger(__name__)

#: Optional (None-default) ingest params each kind honours; passing any
#: other one is refused rather than silently dropped.
_INGEST_HONOURS = {
    "md": frozenset({"md_path", "concept_id", "title", "description", "tags"}),
    "thoughts": frozenset({"thoughts", "topic", "concept_id", "tags"}),
    "pdf": frozenset({"pdf_path", "output_dir"}),
}


def _slugify_topic(topic: str) -> str:
    """Map a thought topic to a single safe namespace level.

    Lowercased, spaces to underscores, everything outside
    ``[A-Za-z0-9_.-]`` collapsed to ``_`` — so no ``/`` (nesting) and no
    ``..`` (export path traversal) can ever come from a topic string.
    Returns ``""`` when nothing survives (caller falls back to a default).
    """
    slug = re.sub(r"[^A-Za-z0-9_.\-]+", "_", topic.lower().replace(" ", "_"))
    return slug.strip("._")[:30]


def _require_file(path: str | Path, kind: str = "file") -> None:
    """Raise the typed ``FILE_NOT_FOUND`` error when a path is absent."""
    if not Path(path).exists():
        raise OKFError(
            "FILE_NOT_FOUND",
            f"{kind} not found: {path}",
            fields={"path": str(path)},
        )


def _ingest_namespace(pdf_path, work_dir) -> str:
    """Stable ingest namespace for one PDF auto-import (Phase 2 §2.1).

    `pdf-<hash12>` from the source bytes when readable, else a fingerprint
    of the converted work dir (sorted paths + sizes). Either is stable
    across re-imports of the same source and distinct across sources.
    """
    try:
        with open(pdf_path, "rb") as _f:
            return f"pdf-{hashlib.sha256(_f.read()).hexdigest()[:12]}"
    except OSError:
        pass
    try:
        entries = sorted(
            f"{p.relative_to(work_dir)}:{p.stat().st_size}"
            for p in sorted(Path(work_dir).rglob("*"))
            if p.is_file()
        )
        return f"pdf-{hashlib.sha256(chr(10).join(entries).encode()).hexdigest()[:12]}"
    except OSError:
        return f"pdf-{uuid.uuid4().hex[:12]}"


class IngestManager:
    def __init__(self, _write_lock_ctx, bundle_root, device, import_mgr, delta_mgr,
                 converter=None):
        self._write_lock_ctx = _write_lock_ctx
        self.bundle_root = bundle_root
        self.device = device
        self.import_mgr = import_mgr
        self.delta_mgr = delta_mgr
        # DocumentConverter (see okfgraph.components.converters). None =
        # default BobineConverter, built lazily so router construction
        # never requires bobine — only actual conversion does.
        self._converter = converter

    def _ingest_md_inner(
        self,
        md_path: str | Path,
        concept_id: str | None,
        title: str | None,
        description: str | None,
        tags: list[str] | None,
        mode: str,
        force: bool = False,
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

        # Determine metadata. Without an explicit concept_id, a file inside
        # a named root mints that root's namespaced ID (§2.6); outside every
        # root keeps the legacy bare-stem fallback.
        if concept_id:
            cid = concept_id
        else:
            from okfgraph.components.import_ import parse_source_file
            from okfgraph.components.roots import resolve_alias_for_path
            _alias = resolve_alias_for_path(md_path, self.import_mgr.roots) or ""
            _roots = self.import_mgr.roots
            _root = _roots[_alias] if _alias else self.import_mgr.bundle_root
            cid = parse_source_file(md_path, _root, _alias)[2]
        t = title or fm.get("title") or md_path.stem
        desc = description or fm.get("description") or fm.get("summary") or ""
        file_tags = normalize_tags(fm.get("tags", []))
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
        result = self.import_mgr._import_single_concept(concept, post.content, mode, force)

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
        force: bool = False,
    ) -> Dict[str, Any]:
        """Inner implementation of ingest_thoughts (called under write lock)."""
        import uuid

        # Generate concept_id from topic if not provided. Namespaced
        # (``thoughts/<slug>/...``) so a fileless graph exports into a tidy
        # tree later — the namespace is virtual (no file needed) until an
        # export materializes it. Slashes in an explicit concept_id work
        # the same way. The slug is sanitized to one level: a topic like
        # ``../../evil`` must never become path traversal on export.
        if not concept_id:
            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            slug = _slugify_topic(topic) or "untitled"
            concept_id = f"thoughts/{slug}/{ts}_{str(uuid.uuid4())[:6]}"

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
        result = self.import_mgr._import_single_concept(concept, markdown, "text", force)

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


    def ingest(
        self,
        kind: str,
        *,
        md_path: str | Path | None = None,
        pdf_path: str | Path | None = None,
        thoughts: str | None = None,
        topic: str | None = None,
        concept_id: str | None = None,
        title: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        mode: str = "text",
        routing_mode: str = "auto",
        extract_images: bool = True,
        auto_import: bool = True,
        output_dir: str | Path | None = None,
        batch_size: int = 32,
        prune_missing: bool = False,
        force: bool = False,
        on_page: Callable[[int, int], None] | None = None,
        converter=None,
    ) -> Dict[str, Any]:
        """One ingest dispatch (0.10: five spellings merged into one).

        kind='md': import a markdown file (md_path required).
        kind='pdf': convert (pdf_path required; bobine) and auto-import
        unless ``auto_import=False``, which writes ``output_dir`` only.
        kind='thoughts': persist LLM reasoning (thoughts + topic required).

        File paths that don't exist raise ``FILE_NOT_FOUND``; missing
        required params raise ``MISSING_PARAM``; a passed param the kind
        ignores raises ``BAD_VALUE`` naming it. ``on_page``/``converter``
        are Python-only advanced params.
        """
        passed = {
            "md_path": md_path, "pdf_path": pdf_path, "thoughts": thoughts,
            "topic": topic, "concept_id": concept_id, "title": title,
            "description": description, "tags": tags,
            "output_dir": output_dir,
        }
        honours = _INGEST_HONOURS.get(kind)
        if honours is not None:
            if kind == "pdf" and auto_import:
                honours = honours - {"output_dir"}
            ignored = [k for k, v in passed.items()
                       if v is not None and k not in honours]
            if ignored:
                raise OKFError(
                    "BAD_VALUE",
                    f"kind='{kind}' ignores: {', '.join(ignored)}",
                    fields={"kind": kind, "ignored": ignored},
                    remedy=("output_dir needs auto_import=False"
                            if ignored == ["output_dir"] and kind == "pdf"
                            else "drop the params of other kinds"),
                )
        if kind == "md":
            if not md_path:
                raise OKFError(
                    "MISSING_PARAM",
                    "kind='md' requires md_path",
                    fields={"kind": kind, "md_path": md_path},
                )
            _require_file(md_path, kind="markdown file")
            with self._write_lock_ctx():
                return self._ingest_md_inner(md_path, concept_id, title, description, tags, mode, force)
        if kind == "thoughts":
            if not thoughts or not topic:
                raise OKFError(
                    "MISSING_PARAM",
                    "kind='thoughts' requires thoughts and topic",
                    fields={"kind": kind, "thoughts": bool(thoughts), "topic": bool(topic)},
                )
            with self._write_lock_ctx():
                return self._ingest_thoughts_inner(thoughts, topic, concept_id, tags, force)
        if kind == "pdf":
            if not pdf_path:
                raise OKFError(
                    "MISSING_PARAM",
                    "kind='pdf' requires pdf_path",
                    fields={"kind": kind, "pdf_path": pdf_path},
                )
            _require_file(pdf_path, kind="PDF")
            pdf_path = Path(pdf_path)
            converter = self._resolve_converter(
                converter, routing_mode=routing_mode,
                extract_images=extract_images,
            )
            from tempfile import TemporaryDirectory

            if auto_import:
                with TemporaryDirectory(prefix="okf_ingest_") as tmp:
                    work_dir = Path(tmp)
                    logger.info("converting %s → %s", pdf_path, work_dir)
                    doc = converter.convert(pdf_path, work_dir, on_page=on_page)
                    md_path = Path(doc.md_path)
                    lint_result = self._lint_converted_md(md_path, auto_fix=True)
                    if lint_result["fixed"]:
                        md_path.write_text(lint_result["content"], encoding="utf-8")
                    if lint_result["errors"]:
                        logger.warning(
                            "PDF output has %d structural errors — proceeding anyway",
                            len(lint_result["errors"]),
                        )
                    ids = self._import_work_dir(
                        work_dir, batch_size, mode, prune_missing, pdf_path, force
                    )
                    return {
                        "md_path": str(md_path),
                        "concept_ids": ids,
                        "image_dir": str(doc.image_dir),
                        "page_count": doc.page_count,
                    }
            output_dir = Path(output_dir) if output_dir else pdf_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info("converting %s → %s", pdf_path, output_dir)
            doc = converter.convert(pdf_path, output_dir, on_page=on_page)
            md_path = Path(doc.md_path)
            lint_result = self._lint_converted_md(md_path, auto_fix=True)
            if lint_result["fixed"]:
                md_path.write_text(lint_result["content"], encoding="utf-8")
            logger.info("written %s", md_path)
            return {
                "md_path": str(md_path),
                "concept_ids": [],
                "image_dir": str(doc.image_dir),
                "page_count": doc.page_count,
            }
        raise OKFError(
            "BAD_VALUE",
            f"kind must be 'md', 'pdf' or 'thoughts', got '{kind}'",
            fields={"kind": kind},
        )



    def _resolve_converter(self, converter, routing_mode="auto", extract_images=True):
        """Per-call override → manager default → lazy BobineConverter.

        ``routing_mode``/``extract_images`` only apply when a BobineConverter
        is built here; an explicit ``converter`` instance is used verbatim.
        """
        if converter is not None:
            return converter
        if (routing_mode != "auto" or not extract_images):
            # Non-default converter knobs: build ad hoc, never cache —
            # a cached converter would pin one call's routing for all.
            from okfgraph.components.converters import BobineConverter
            return BobineConverter(routing_mode=routing_mode,
                                   extract_images=extract_images)
        if self._converter is None:
            from okfgraph.components.converters import BobineConverter
            self._converter = BobineConverter()
        return self._converter

    def _import_work_dir(self, work_dir, batch_size, mode, prune_missing, pdf_path, force=False):
        """Import a converted-PDF work dir, keeping bundle_root overrides in sync."""
        from okfgraph.components.delta import DeltaDetector as _DD
        old_bundle_root = self.bundle_root
        self.bundle_root = work_dir
        # Keep the injected DeltaDetector and ImportManager in sync:
        # each stores its own bundle_root copy, and import_bundle /
        # _changed_directories rely on it (Phase 3 refactor).
        self.delta_mgr.bundle_root = work_dir
        self.import_mgr.bundle_root = work_dir
        # Ingest namespace (§2.1): PDF pages mint `@pdf-<hash12>/...` IDs —
        # stable per source (content hash), so same-stem pages from
        # different PDFs never overwrite each other, and re-imports upsert
        # idempotently. The ephemeral detector is suspended with the rest
        # (no baseline writes) and unregistered afterwards.
        _ns = _ingest_namespace(pdf_path, work_dir)
        _mgr = self.import_mgr
        _mgr._delta_by_alias[_ns] = _DD(_mgr.conn, work_dir, _ns)
        try:
            # Suspended (0.2.15): a TemporaryDirectory must never read or
            # write the shared delta baseline — its root-relative keys
            # would collide with the real bundle's (notably top-level ".").
            # Work-dir imports are always full imports. The ephemeral
            # namespace detector suspends too: it must neither persist
            # namespaced keys into the shared tables nor consume tombstones.
            with self.delta_mgr.suspended(), \
                    _mgr._delta_by_alias[_ns].suspended():
                ids = self.import_mgr.import_bundle(
                    work_dir,
                    batch_size=batch_size,
                    mode=mode,
                    prune_missing=prune_missing,
                    alias=_ns,
                    force=force,
                )
        finally:
            self.bundle_root = old_bundle_root
            self.delta_mgr.bundle_root = old_bundle_root
            self.import_mgr.bundle_root = old_bundle_root
            self.import_mgr._delta_by_alias.pop(_ns, None)
        logger.info("imported %d concept(s) from %s", len(ids), pdf_path)
        return ids

