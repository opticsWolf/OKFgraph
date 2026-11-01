from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from datetime import datetime, timezone
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
    "md": frozenset({"md_path", "concept_id", "title", "description", "tags",
                     "type", "extra", "materialize"}),
    "thoughts": frozenset({"thoughts", "topic", "concept_id", "title", "description",
                           "tags", "type", "extra", "materialize", "related",
                           "folder"}),
    "pdf": frozenset({"pdf_path", "output_dir"}),
    # Per-entry IDs come from citation keys, so bib honours only the
    # file-level inputs: one file in, N concepts out.
    "bib": frozenset({"bib_path", "tags"}),
}


#: Concept fields ``extra`` may not set: they have their own parameters or
#: are owned by the pipeline (0.11.0, O9).
_RESERVED_EXTRA = frozenset({
    "id", "type", "title", "description", "resource", "tags", "timestamp",
    "body", "embedding", "uid", "source_path",
})

_SAFE_ID_RE = re.compile(r"^(@[A-Za-z0-9_-]+/)?[A-Za-z0-9_][A-Za-z0-9_.\-]*(/[A-Za-z0-9_][A-Za-z0-9_.\-]*)*$")


def _check_extra(extra) -> Dict[str, Any]:
    """Validate ``extra`` (O9): a mapping of plain keys, none reserved."""
    if extra is None:
        return {}
    if not isinstance(extra, dict):
        raise OKFError("BAD_VALUE", "extra must be a mapping of key -> value",
                       fields={"extra": type(extra).__name__})
    bad = sorted(k for k in extra if not isinstance(k, str) or k in _RESERVED_EXTRA)
    if bad:
        raise OKFError(
            "BAD_VALUE",
            f"extra may not set {', '.join(map(str, bad))} (use the dedicated "
            "parameters: concept_id, type, title, description, tags)",
            fields={"reserved": [str(b) for b in bad]},
        )
    return dict(extra)


#: Relation kind a thought's ``related`` ids are written under.
RELATED = "related"


def _with_related(extra: Dict[str, Any], related) -> Dict[str, Any]:
    """Merge ``related`` ids into ``extra["relations"]["related"]``.

    A thought links to the concepts it is about as typed ``related``
    edges (O6), so traverse can walk from it and PPR/hub see it. Targets
    resolve like any relation: an id, a path, or a wiki name.
    """
    if related is None:
        return extra
    if isinstance(related, str) or not isinstance(related, (list, tuple)):
        raise OKFError("BAD_VALUE", "related must be a list of concept ids",
                       fields={"related": type(related).__name__})
    bad = [r for r in related if not isinstance(r, str) or not r.strip()]
    if bad:
        raise OKFError("BAD_VALUE", "related entries must be non-empty strings",
                       fields={"related": [repr(b) for b in bad]})
    rels = extra.get("relations")
    if rels is None:
        rels = {}
    elif not isinstance(rels, dict):
        raise OKFError("BAD_VALUE", "extra['relations'] must be a mapping of kind -> targets",
                       fields={"relations": type(rels).__name__})
    have = rels.get(RELATED) or []
    if isinstance(have, str):
        have = [have]
    merged = list(dict.fromkeys([*have, *(r.strip() for r in related)]))
    return {**extra, "relations": {**rels, RELATED: merged}}


def _check_concept_id(concept_id: str) -> None:
    """An explicit id that becomes a file path must be a safe relative id."""
    if ".." in concept_id.split("/") or not _SAFE_ID_RE.match(concept_id):
        raise OKFError(
            "BAD_VALUE",
            f"concept_id {concept_id!r} cannot be materialized as a bundle file "
            "(letters, digits, '_', '.', '-' and '/' only; no '..'; an "
            "optional leading @alias/ names a root)",
            fields={"concept_id": concept_id},
        )


def _folder_thought_id(folder: str, title: str | None, concept_id: str | None,
                       materialize: bool | None) -> str:
    """The concept id a thought gets from ``folder`` (0.11.0).

    ``folder`` is a bundle sub-path (``@alias/...`` for another root); the
    file name is the title as a slug, else a timestamp + short uuid so
    untitled thoughts never collide. It replaces ``concept_id`` +
    ``materialize``, so either one beside it is refused.
    """
    import uuid

    if concept_id is not None:
        raise OKFError(
            "BAD_VALUE", "folder and concept_id both set where the thought goes",
            fields={"folder": folder, "concept_id": concept_id},
            remedy="pass folder (file named from the title) or concept_id, not both")
    if materialize is False:
        raise OKFError(
            "BAD_VALUE", "folder writes a file: it cannot run with materialize=False",
            fields={"folder": folder, "materialize": False},
            remedy="drop materialize, or drop folder for a graph-only thought")
    base = str(folder).strip().replace("\\", "/").strip("/")
    name = re.sub(r"[^a-z0-9_.\-]+", "-", (title or "").strip().lower()).strip("-._")[:60]
    if not name:
        name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{str(uuid.uuid4())[:6]}"
    cid = f"{base}/{name}" if base else name
    if (not base or ".." in base.split("/") or str(folder).strip().startswith("/")
            or not _SAFE_ID_RE.match(cid)):
        raise OKFError(
            "BAD_VALUE",
            f"folder {folder!r} is not a bundle sub-path (letters, digits, '_', "
            "'.', '-' and '/' only; no '..', not absolute; @alias/ names a root)",
            fields={"folder": folder})
    return cid


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


def _bib_entry_concept(entry, concept_id: str,
                       file_tags: list[str] | None) -> tuple:
    """Map one parsed BibTeX entry to (ConceptModel, markdown body)."""
    from okfgraph.components.bibtex import unwrap_braces

    f = entry.fields
    title = unwrap_braces(f.get("title", "")).strip() or entry.key
    authors = f.get("author", "").strip()
    year = f.get("year", "").strip()
    venue = next(
        (f.get(k, "").strip()
         for k in ("journal", "booktitle", "school", "publisher")
         if f.get(k, "").strip()),
        "",
    )
    head = " ".join(p for p in [authors, f"({year})" if year else ""] if p)
    citation = f"{head}. {venue}".strip(" .") if (head or venue) else entry.key
    entry_tags = ["reference", entry.entrytype] + ([year] if year else [])
    all_tags = normalize_tags(entry_tags + list(file_tags or []))
    metadata: Dict[str, Any] = {
        "title": title,
        "type": "reference",
        "citekey": entry.key,
        "entrytype": entry.entrytype,
    }
    for meta_key, bib_key in (("author", "author"), ("year", "year"),
                              ("venue", None), ("doi", "doi"),
                              ("url", "url")):
        value = venue if bib_key is None else f.get(bib_key, "").strip()
        if value:
            metadata[meta_key] = value
    lines = ["## Citation", "", citation, ""]
    links = []
    if f.get("doi", "").strip():
        doi = f["doi"].strip()
        links.append(f"[DOI](https://doi.org/{doi})")
    if f.get("url", "").strip():
        links.append(f"[URL]({f['url'].strip()})")
    if links:
        lines += [" ".join(links), ""]
    abstract = unwrap_braces(f.get("abstract", "")).strip()
    lines += ["## Abstract", "", abstract or "_No abstract provided._", ""]
    markdown = frontmatter.dumps(frontmatter.Post("\n".join(lines), **metadata))
    concept = ConceptModel.model_validate({
        "id": concept_id,
        "title": title,
        "description": citation,
        "body": markdown,
        "type": "reference",
        "tags": all_tags,
    })
    return concept, markdown


#: Auto-titles for thoughts are cut to this many characters (WS2).
THOUGHT_TITLE_MAX = 72


def _auto_thought_title(text: str, topic: str, now: datetime) -> str:
    """Title for a thought with none given: its first sentence.

    Whitespace is collapsed, the sentence is cut at a word boundary to at most
    :data:`THOUGHT_TITLE_MAX` characters, and trailing ``.!?`` is dropped. If
    the text has no usable sentence (fewer than three letters or digits), the
    title is ``<topic> — <YYYY-MM-DD HH:MM>``.
    """
    flat = " ".join((text or "").split())
    m = re.match(r"(.+?[.!?])(?:\s|$)", flat)
    sentence = (m.group(1) if m else flat).rstrip(".!? ")
    if sum(ch.isalnum() for ch in sentence) < 3:
        return f"{topic} — {now:%Y-%m-%d %H:%M}"
    if len(sentence) > THOUGHT_TITLE_MAX:
        cut = sentence[:THOUGHT_TITLE_MAX]
        sentence = cut.rsplit(" ", 1)[0] if " " in cut else cut
    return sentence


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
        type: str | None = None,
        extra: Dict[str, Any] | None = None,
        materialize: bool = False,
    ) -> Dict[str, Any]:
        """Inner implementation of ingest_md (called under write lock).

        0.11.0: the file is parsed exactly as a bundle import parses it, so
        every frontmatter key (``relations:``, Jev fields, ...) survives —
        before, md ingest kept only title/description/type/tags. Explicit
        params override frontmatter; ``extra`` merges over it. When the file
        lives in a configured root its hash is recorded (O2), so the next
        bundle import does not embed it again. ``materialize`` (O9) writes
        the resulting markdown into the bundle at ``<concept_id>.md`` and
        imports that file — for a source outside every root.
        """
        md_path = Path(md_path)
        if not md_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {md_path}")

        # Lint report only: md_path is the user's source, never rewritten
        # (bulk import doesn't touch sources either, and a fix would make
        # the graph drift from the file). Fixable issues are only counted.
        lint_result = self._lint_converted_md(md_path, auto_fix=False)

        from okfgraph.components.import_ import parse_source_file
        from okfgraph.components.roots import resolve_alias_for_path
        _alias = resolve_alias_for_path(md_path, self.import_mgr.roots) or ""
        _roots = self.import_mgr.roots
        _root = _roots[_alias] if _alias else self.import_mgr.bundle_root
        concept, body, path_cid = parse_source_file(
            md_path, _root, _alias, identity=self.import_mgr.identity)
        cid = concept_id or path_cid

        data = concept.model_dump()
        data.pop("embedding", None)
        data["id"] = cid
        if title:
            data["title"] = title
        if description:
            data["description"] = description
        elif not data.get("description"):
            data["description"] = data.get("summary") or ""
        if type:
            data["type"] = type
        data["tags"] = sorted(set((tags or []) + normalize_tags(data.get("tags"))))
        data.update(_check_extra(extra))
        concept = ConceptModel.model_validate(data)

        if materialize:
            return self._materialize_and_import(concept, body, mode, force,
                                                lint_result=lint_result)

        inside = False
        if _root is not None:
            try:
                inside = md_path.resolve().is_relative_to(Path(_root).resolve())
            except (OSError, ValueError):
                inside = False

        # Import via shared single-concept pipeline
        result = self.import_mgr._import_single_concept(
            concept, body, mode, force, base_dir=md_path.parent,
            source_file=md_path if inside else None)

        return {
            "concept_id": result["concept_id"],
            "title": result["title"],
            "description": result["description"],
            "tags": result["tags"],
            "chunk_count": result["chunk_count"],
            "image_count": result["image_count"],
            "lint_issues": {
                "fixed_count": lint_result["fixed_count"],
                "fixable_count": lint_result["fixable_count"],
                "unfixable_count": len(lint_result["unfixable"]),
                "error_count": len(lint_result["errors"]),
            },
        }

    def _bundle_file_for(self, concept_id: str) -> Path:
        """Where ``concept_id`` lives as a bundle file (O9 materialize).

        ``@alias/rest`` lands in the named root, anything else in the
        primary root. File-free routers cannot materialize.
        """
        _check_concept_id(concept_id)
        roots = self.import_mgr.roots
        if concept_id.startswith("@"):
            alias, rest = concept_id[1:].split("/", 1)
            root = roots.get(alias)
            if root is None:
                raise OKFError(
                    "BAD_VALUE", f"unknown root alias @{alias} in concept_id",
                    fields={"concept_id": concept_id, "roots": sorted(roots)})
        else:
            root, rest = self.import_mgr.bundle_root, concept_id
        if root is None:
            raise OKFError(
                "MISSING_PARAM",
                "materialize needs a bundle root: this router is file-free "
                "(bundle_root=None)",
                fields={"materialize": True},
                remedy="open the router with bundle_root, or omit materialize",
            )
        # Under frontmatter identity the id also travels in the file's
        # frontmatter; the file still lands at the id's path, so people
        # find it where the id says.
        return Path(root) / f"{rest}.md"

    def _materialize_and_import(self, concept: ConceptModel, body: str,
                                mode: str, force: bool,
                                lint_result=None) -> Dict[str, Any]:
        """Write ``concept`` as a bundle file, then import that file (O9).

        The single-writer rule: the bundle stays the source of truth and the
        graph is what importing it produced. Re-materializing the same id
        overwrites the same file — the same call twice is one concept.
        """
        path = self._bundle_file_for(concept.id)
        fm, _ = concept.export_frontmatter()
        fm.pop("source_path", None)
        if self.import_mgr.identity == "frontmatter":
            raw_id = concept.id.split("/", 1)[1] if concept.id.startswith("@") else concept.id
            fm["id"] = raw_id
        else:
            fm.pop("id", None)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(frontmatter.dumps(frontmatter.Post(body, **fm)) + "\n",
                        encoding="utf-8")
        cid = self.import_mgr.import_from_okf(path, mode=mode, force=force)
        out = {
            "concept_id": cid,
            "path": str(path),
            "materialized": True,
            "title": concept.title,
            "tags": concept.tags,
        }
        if lint_result is not None:
            out["lint_issues"] = {
                "fixed_count": lint_result["fixed_count"],
                "fixable_count": lint_result["fixable_count"],
                "unfixable_count": len(lint_result["unfixable"]),
                "error_count": len(lint_result["errors"]),
            }
        return out


    def _thought_titles_in_topic(self, slug: str, title: str, exclude_id: str) -> list[str]:
        """Ids of other thoughts in the topic namespace already titled ``title``.

        Scope is the topic: default ids are ``thoughts/<slug>/...``, so the
        namespace prefix is the topic. Ids outside it are not compared.
        """
        rows = self.import_mgr.conn.execute(
            "MATCH (c:Concept) WHERE c.title = $t "
            "RETURN c.id AS id",
            {"t": title},
        ).rows_as_dict().get_all() or []
        prefix = f"thoughts/{slug}/"
        return [r["id"] for r in rows
                if r["id"] != exclude_id and r["id"].startswith(prefix)]

    def retitle_thoughts(self, topic: str | None = None,
                         apply: bool = False) -> Dict[str, Any]:
        """Give old-style ``Thought: <topic>`` titles a real one (WS2, Q3).

        Each thought still titled exactly ``Thought: <its topic>`` gets the
        title a new thought would get: its first sentence, with ``(2)``,
        ``(3)`` on clashes within the topic. ``topic`` limits the run to one
        topic. Without ``apply`` nothing is written and the plan is
        returned; with it, each thought is rewritten under its own id (type,
        tags, description, extra frontmatter and ``created`` kept,
        re-embedded). A materialized thought (O9) is backed by a bundle
        file, so it is listed under ``skipped`` instead: edit its file.
        """
        rows = self.import_mgr.conn.execute(
            "MATCH (c:Concept) WHERE c.title STARTS WITH 'Thought: ' "
            "RETURN c.id AS id, c.title AS title, c.type AS type, "
            "c.description AS description, c.tags AS tags, c.body AS body "
            "ORDER BY c.id"
        ).rows_as_dict().get_all() or []
        # A FileHash row means the concept came from a bundle file (O9).
        file_backed = {
            r["c"] for r in self.import_mgr.conn.execute(
                "MATCH (f:FileHash) WHERE f.concept_id IN $ids "
                "RETURN f.concept_id AS c",
                {"ids": [r["id"] for r in rows]},
            ).rows_as_dict().get_all()
        } if rows else set()
        plan: list[Dict[str, Any]] = []
        work: list[Dict[str, Any]] = []
        skipped: list[str] = []
        planned: Dict[str, set] = {}  # slug -> titles given out in this run
        for r in rows:
            post = frontmatter.loads(r["body"] or "")
            if r["type"] != "thought" and post.metadata.get("thought_type") != "reasoning":
                continue  # not a thought
            t = str(post.metadata.get("topic") or r["title"][len("Thought: "):])
            if r["title"] != f"Thought: {t}":
                continue  # a deliberate title that happens to start that way
            if topic is not None and t != topic:
                continue
            if r["id"] in file_backed:
                skipped.append(r["id"])
                continue
            created = post.metadata.get("created")
            if isinstance(created, str):
                try:
                    created = datetime.fromisoformat(created)
                except ValueError:
                    created = None
            if not isinstance(created, datetime):
                created = None
            slug = _slugify_topic(t) or "untitled"
            taken = planned.setdefault(slug, set())
            base = _auto_thought_title(post.content, t, created or datetime.now())
            new, n = base, 2
            while new in taken or self._thought_titles_in_topic(slug, new, r["id"]):
                new = f"{base} ({n})"
                n += 1
            taken.add(new)
            plan.append({"concept_id": r["id"], "topic": t,
                         "old_title": r["title"], "new_title": new})
            work.append({"text": post.content, "topic": t, "id": r["id"],
                         "tags": [x for x in (r["tags"] or [])
                                  if x not in ("thought", "reasoning", t)],
                         "title": new, "description": r["description"],
                         "type": r["type"], "created": created,
                         "extra": {k: v for k, v in post.metadata.items()
                                   if k not in _RESERVED_EXTRA
                                   and k not in ("thought_type", "topic", "created")}})
        if apply and work:
            with self._write_lock_ctx():
                for w in work:
                    self._ingest_thoughts_inner(
                        w["text"], w["topic"], w["id"], w["tags"],
                        title=w["title"], description=w["description"],
                        type=w["type"], extra=w["extra"], created=w["created"])
        return {"applied": bool(apply), "count": len(plan), "plan": plan,
                "skipped": skipped}

    def _ingest_thoughts_inner(
        self,
        thoughts: str,
        topic: str,
        concept_id: str | None,
        tags: list[str] | None,
        force: bool = False,
        title: str | None = None,
        description: str | None = None,
        type: str | None = None,
        extra: Dict[str, Any] | None = None,
        materialize: bool = False,
        created: datetime | None = None,
        related: list[str] | None = None,
    ) -> Dict[str, Any]:
        """Inner implementation of ingest_thoughts (called under write lock).

        0.11.0 (O9): ``type`` (default ``thought``) and ``extra`` are stored
        like frontmatter; an explicit ``concept_id`` upserts (the same call
        twice is one concept, links into it kept — O1); ``materialize``
        writes the thought into the bundle and imports the file.
        ``related`` adds typed ``related`` links to those concepts.
        ``created`` keeps an existing thought's timestamp on a rewrite
        (retitle); None = now.
        """
        import uuid

        now = created or datetime.now()
        slug = _slugify_topic(topic) or "untitled"
        # Generate concept_id from topic if not provided. Namespaced
        # (``thoughts/<slug>/...``) so a fileless graph exports into a tidy
        # tree later — the namespace is virtual (no file needed) until an
        # export materializes it. Slashes in an explicit concept_id work
        # the same way. The slug is sanitized to one level: a topic like
        # ``../../evil`` must never become path traversal on export.
        if not concept_id:
            ts = now.strftime("%Y%m%d%H%M%S")
            concept_id = f"thoughts/{slug}/{ts}_{str(uuid.uuid4())[:6]}"

        # Titles are unique per topic (WS2). An explicit title is a name
        # someone will link to, so a clash is refused, never renamed. An
        # auto-title gets a (2), (3), … suffix instead.
        title = (title or "").strip() or None
        if title:
            taken = self._thought_titles_in_topic(slug, title, concept_id)
            if taken:
                raise OKFError(
                    "BAD_VALUE",
                    f"a thought titled {title!r} already exists in topic {topic!r}: {taken[0]}",
                    fields={"title": title, "topic": topic, "existing_id": taken[0]},
                    remedy="choose another title, or pass --concept-id "
                           f"{taken[0]} to update that thought",
                )
        else:
            base = _auto_thought_title(thoughts, topic, now)
            title, n = base, 2
            while self._thought_titles_in_topic(slug, title, concept_id):
                title = f"{base} ({n})"
                n += 1

        extra = _with_related(_check_extra(extra), related)
        ctype = type or "thought"
        # Build OKF-compliant markdown (frontmatter via the YAML dumper, so
        # extra values with ':' or quotes cannot break the header).
        meta: Dict[str, Any] = {
            "title": title,
            "type": ctype,
            "thought_type": "reasoning",
            "topic": topic,
            "tags": ["thought", "reasoning", topic],
            "created": now.isoformat(),
        }
        meta.update(extra)
        markdown = frontmatter.dumps(frontmatter.Post(thoughts, **meta))

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
            **extra,
            "id": concept_id,
            "title": title,
            "description": (description or "").strip() or f"Reasoning about {topic}",
            "body": markdown,
            "type": ctype,
            "tags": all_tags,
            # Dated by when it was first written (kept on a rewrite), in
            # the naive-UTC form every TIMESTAMP column uses.
            "timestamp": now.astimezone(timezone.utc).replace(tzinfo=None),
        })

        if materialize:
            body = frontmatter.loads(markdown).content
            data = concept.model_dump()
            data.pop("embedding", None)
            data.update({"body": body, "topic": topic,
                         "thought_type": "reasoning",
                         "created": meta["created"]})
            concept = ConceptModel.model_validate(data)
            out = self._materialize_and_import(concept, body, "text", force)
            out.update({"topic": topic, "markdown": markdown,
                        "lint_issues": lint_result})
            if related:
                out["unresolved_related"] = self._unresolved_related(out["concept_id"])
            return out

        # Import via shared single-concept pipeline
        result = self.import_mgr._import_single_concept(concept, markdown, "text", force)

        out = {
            "concept_id": result["concept_id"],
            "title": title,
            "topic": topic,
            "tags": result["tags"],
            "chunk_count": result["chunk_count"],
            "markdown": markdown,
            "lint_issues": lint_result,
        }
        if related:
            out["unresolved_related"] = self._unresolved_related(result["concept_id"])
        return out

    def _unresolved_related(self, concept_id: str) -> list[str]:
        """``related`` targets of ``concept_id`` that resolved to no concept
        (kept as BrokenLink rows, repaired once the target exists)."""
        rows = self.import_mgr.conn.execute(
            "MATCH (bl:BrokenLink) WHERE bl.source_id = $s AND bl.kind = $k "
            "RETURN bl.raw AS raw ORDER BY bl.raw",
            {"s": concept_id, "k": RELATED},
        ).rows_as_dict().get_all()
        return [r["raw"] for r in rows]

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
        - "fixable_count": fixable issues found, fixed or not (int)
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
                "fixable_count": 0,
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
            "fixable_count": len(diagnostics) - len(unfixable),
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


    def _ingest_bib_inner(
        self,
        bib_path: str | Path,
        tags: list[str] | None,
        mode: str,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Inner implementation of kind='bib' (called under write lock)."""
        from okfgraph.components.bibtex import parse_bibtex, slugify_key

        bib_path = Path(bib_path)
        if not bib_path.exists():
            raise FileNotFoundError(f"BibTeX file not found: {bib_path}")
        try:
            text = bib_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning(
                "ingest_bib: %s is not UTF-8, falling back to latin-1",
                bib_path,
            )
            text = bib_path.read_text(encoding="latin-1")
        parsed = parse_bibtex(text)
        skipped: List[Dict[str, str]] = list(parsed.skipped)
        if not parsed.entries:
            raise OKFError(
                "BAD_VALUE",
                f"no usable BibTeX entries in {bib_path}",
                fields={"bib_path": str(bib_path), "skipped": skipped},
            )
        counts: Dict[str, int] = {}
        first_key: Dict[str, str] = {}
        ids: List[str] = []
        for entry in parsed.entries:
            slug = slugify_key(entry.key)
            if slug in counts:
                counts[slug] += 1
                cid = f"refs/{slug}~{counts[slug]}"
                skipped.append({
                    "key": entry.key, "line": str(entry.line),
                    "concept_id": cid,
                    "reason": f"slug collides with '{first_key[slug]}'; "
                                "disambiguated, nothing overwritten",
                })
                logger.warning(
                    "ingest_bib: key '%s' slug-collides, minted %s",
                    entry.key, cid,
                )
            else:
                counts[slug] = 1
                first_key[slug] = entry.key
                cid = f"refs/{slug}"
            concept, markdown = _bib_entry_concept(entry, cid, tags)
            lint_result = self._lint_converted_md_str(markdown, auto_fix=True)
            if lint_result["fixed"]:
                markdown = lint_result["content"]
            if lint_result["errors"]:
                logger.warning(
                    "ingest_bib: %s has %d structural errors — "
                    "importing anyway", cid, len(lint_result["errors"]),
                )
            result = self.import_mgr._import_single_concept(
                concept, markdown, mode, force)
            ids.append(result["concept_id"])
        return {
            "concept_ids": ids,
            "entry_count": len(ids),
            "skipped_entries": skipped,
        }

    def ingest(
        self,
        kind: str,
        *,
        md_path: str | Path | None = None,
        pdf_path: str | Path | None = None,
        bib_path: str | Path | None = None,
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
        type: str | None = None,
        extra: Dict[str, Any] | None = None,
        materialize: bool | None = None,
        rebuild: str = "auto",
        related: list[str] | None = None,
        folder: str | None = None,
    ) -> Dict[str, Any]:
        """One ingest dispatch (0.10: five spellings merged into one).

        kind='md': import a markdown file (md_path required).
        kind='pdf': convert (pdf_path required; bobine) and auto-import
        unless ``auto_import=False``, which writes ``output_dir`` only.
        kind='thoughts': persist LLM reasoning (thoughts + topic required).
        kind='bib': import one concept per BibTeX entry (bib_path
        required); IDs are ``refs/<key>``, stable across re-imports.

        0.11.0 (O9): ``type`` and ``extra`` (md, thoughts) store a concept
        type and frontmatter-like fields; ``materialize=True`` writes the
        markdown into the bundle and imports the file (single-writer rule).
        ``rebuild`` (O8: auto | defer | force) is the index step afterwards.
        ``related`` (thoughts) links the thought to those concepts as typed
        ``related`` relations (O6). ``folder`` (thoughts) writes the thought
        as ``<folder>/<title-slug>.md`` in the bundle and imports it: the
        same as ``concept_id`` + ``materialize=True``, minus naming the file.

        File paths that don't exist raise ``FILE_NOT_FOUND``; missing
        required params raise ``MISSING_PARAM``; a passed param the kind
        ignores raises ``BAD_VALUE`` naming it. ``on_page``/``converter``
        are Python-only advanced params.
        """
        passed = {
            "md_path": md_path, "pdf_path": pdf_path, "thoughts": thoughts,
            "topic": topic, "concept_id": concept_id, "title": title,
            "description": description, "tags": tags,
            "output_dir": output_dir, "bib_path": bib_path,
            "type": type, "extra": extra, "materialize": materialize,
            "related": related, "folder": folder,
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
            with self._write_lock_ctx(), self.import_mgr.rebuild(rebuild):
                return self._ingest_md_inner(
                    md_path, concept_id, title, description, tags, mode, force,
                    type=type, extra=extra, materialize=bool(materialize))
        if kind == "bib":
            if not bib_path:
                raise OKFError(
                    "MISSING_PARAM",
                    "kind='bib' requires bib_path",
                    fields={"kind": kind, "bib_path": bib_path},
                )
            _require_file(bib_path, kind="BibTeX file")
            with self._write_lock_ctx():
                return self._ingest_bib_inner(bib_path, tags, mode, force)
        if kind == "thoughts":
            if not thoughts or not topic:
                raise OKFError(
                    "MISSING_PARAM",
                    "kind='thoughts' requires thoughts and topic",
                    fields={"kind": kind, "thoughts": bool(thoughts), "topic": bool(topic)},
                )
            if folder is not None:
                concept_id = _folder_thought_id(folder, title, concept_id, materialize)
                materialize = True
            with self._write_lock_ctx(), self.import_mgr.rebuild(rebuild):
                return self._ingest_thoughts_inner(
                    thoughts, topic, concept_id, tags, force,
                    title=title, description=description,
                    type=type, extra=extra, materialize=bool(materialize),
                    related=related)
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
            f"kind must be 'md', 'pdf', 'thoughts' or 'bib', got '{kind}'",
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

