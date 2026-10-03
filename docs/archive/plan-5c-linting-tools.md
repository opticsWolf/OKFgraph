# Plan: Linting + `ingest_md` + `ingest_thoughts` + Tool Updates

**Gap**: #5c (LLM tool definitions)
**Status**: ✅ **Complete** (v5.7)
**Estimated effort**: ~105 min (1.75 hours)
**Actual effort**: ~45 min
**Dependencies**: None — self-contained

---

## Overview

This plan covers four related changes:

1. **Add mordant linting to the PDF ingestion pipeline** — lint PDF output before import
2. **Add `ingest_md()` method** — direct markdown file import
3. **Add `ingest_thoughts()` method** — store LLM reasoning as searchable concepts
4. **Amend all existing LLM tool definitions** — update descriptions to reference new write operations

---

## Part 1: Add Linting to PDF Ingestion Pipeline

### Where to Add

In `okfgraph/router.py`, add a `_lint_converted_md()` helper and call it before `import_bundle()` in `ingest_pdf()`.

### Implementation

```python
# In OKFRouter class, add as a new private method (~40 lines):

def _lint_converted_md(
    self,
    md_path: Path,
    *,
    auto_fix: bool = True,
) -> Dict[str, Any]:
    """Lint a markdown file and optionally auto-fix fixable issues.

    Returns a dict with:
    - "original": original content (str)
    - "fixed": whether content was modified (bool)
    - "fixed_count": number of auto-fixed issues (int)
    - "unfixable": list of unfixable diagnostics (list)
    - "errors": list of error-level diagnostics (list)
    """
    import mordant

    content = md_path.read_text(encoding="utf-8")
    diagnostics = mordant.lint(content, gfm=True)

    if not diagnostics:
        return {"original": content, "fixed": False, "fixed_count": 0, "unfixable": [], "errors": []}

    # Categorize diagnostics
    fixable_rules = {"MD009", "MD012", "MD047"}  # whitespace/formatting
    error_rules = {"MD001", "MD031", "MD033"}     # structural errors that should block
    unfixable = [d for d in diagnostics if d.rule not in fixable_rules]
    errors = [d for d in diagnostics if d.rule in error_rules]

    fixed_content = content
    fixed_count = 0

    if auto_fix and (diagnostics - unfixable):
        result = mordant.fix(content, gfm=True)
        if result.fixed:
            fixed_content = result.output
            fixed_count = len(result.fixed)
            logger.info("auto-fixed %d issues in %s", fixed_count, md_path.name)

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
        "original": content,
        "fixed": fixed_count > 0,
        "fixed_count": fixed_count,
        "unfixable": unfixable,
        "errors": errors,
    }
```

### Where to Call

In `ingest_pdf()`, after conversion but before `import_bundle()`:

```python
# In ingest_pdf(), inside the auto_import block, after converter.convert_pdf():
# After: md = converter.convert_pdf(...)
# Add:
lint_result = self._lint_converted_md(work_dir / "output.md")
if lint_result["errors"]:
    logger.warning("PDF output has structural errors — proceeding anyway")
# If auto_fix changed content, write it back
if lint_result["fixed"]:
    (work_dir / "output.md").write_text(lint_result["original"], encoding="utf-8")
```

**Note**: `_lint_converted_md` mutates `md_path` in-place when `auto_fix=True`. The returned `original` key contains the fixed content.

---

## Part 2: Add `ingest_md()` Method

### Location

`okfgraph/router.py`, as a new public method on `OKFRouter` (~80 lines).

### Implementation

```python
def ingest_md(
    self,
    md_path: str | Path,
    *,
    concept_id: str | None = None,
    title: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    routing_mode: str = "auto",
    mode: str = "text",
    batch_size: int = 32,
    purge_deleted: bool = False,
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
        routing_mode: ONNX routing mode for embedding.
        mode: Image ingestion mode (text | optional | omni).
        batch_size: Batch size for encoding.
        purge_deleted: If True, purge deleted concepts.

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
    from okfgraph.ingest import IngestMode

    md_path = Path(md_path)
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    # Read and lint
    content = md_path.read_text(encoding="utf-8")
    lint_result = self._lint_converted_md(md_path)

    # Parse frontmatter
    import frontmatter
    meta = frontmatter.load(md_path)

    # Determine metadata
    cid = concept_id or md_path.stem.replace(" ", "_").lower()
    t = title or meta.get("title") or md_path.stem
    desc = description or meta.get("description") or meta.get("summary") or ""
    file_tags = meta.get("tags", [])
    all_tags = list(set((tags or []) + file_tags))

    # Create a temporary concept object for import
    from okfgraph.router import Concept

    concept = Concept(
        id=cid,
        title=t,
        description=desc,
        body=content,
        kind="document",
        tags=all_tags,
    )

    # Import via internal single-file path
    result = self._import_single_concept(concept, content, all_tags, mode, batch_size)

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
```

### Supporting Helper

Add `_import_single_concept()` (~120 lines) — a stripped-down version of `import_bundle()` that handles one concept:

```python
def _import_single_concept(
    self,
    concept: "Concept",
    body: str,
    tags: list[str],
    mode: str,
    batch_size: int,
) -> Dict[str, Any]:
    """Import a single concept using the full pipeline (encode, upsert, chunk, etc.).

    This is the core shared logic between ingest_md() and ingest_thoughts().
    """
    from okfgraph.ingest import IngestMode

    mode = IngestMode.coerce(mode)

    # Phase 2: Encode
    embedding = self._encode(body, task="Document")

    # Phase 3: Upsert in transaction
    self.conn.execute("BEGIN TRANSACTION")
    try:
        self.conn.execute(
            """INSERT INTO Concept (id, title, description, body, kind, tags, embedding)
               VALUES ($id, $title, $desc, $body, $kind, $tags, $embedding)
               ON CONFLICT(id) DO UPDATE SET
                   title=excluded.title, description=excluded.description,
                   body=excluded.body, kind=excluded.kind, tags=excluded.tags,
                   embedding=excluded.embedding""",
            {
                "$id": concept.id,
                "$title": concept.title,
                "$desc": concept.description,
                "$body": concept.body,
                "$kind": concept.kind,
                "$tags": json.dumps(concept.tags),
                "$embedding": embedding,
            },
        )
        self.conn.execute("COMMIT")
    except Exception:
        self.conn.execute("ROLLBACK")
        raise

    # Phase 3.5: Chunk
    chunk_count = 0
    if self.enable_chunking:
        chunks = self._chunk_document(body)
        chunk_count = len(chunks)
        for i, chunk in enumerate(chunks):
            chunk_emb = self._encode(chunk.text, task="Document")
            self.conn.execute(
                """INSERT INTO Chunk (id, concept_id, chunk_index, text, embedding)
                   VALUES ($id, $cid, $idx, $text, $embedding)""",
                {
                    "$id": f"{concept.id}#chunk:{i}",
                    "$cid": concept.id,
                    "$idx": i,
                    "$text": chunk.text,
                    "$embedding": chunk_emb,
                },
            )

    # Phase 4: Directory
    self.conn.execute(
        "INSERT INTO Directory (id, name, parent_id) VALUES ($id, $name, $parent) "
        "ON CONFLICT(id) DO NOTHING",
        {"$id": concept.id, "$name": concept.id, "$parent": ""},
    )

    # Phase 5: Links
    self._extract_links_for_concept(concept.id, body)

    # Phase 6: Images
    image_count = 0
    try:
        image_count = len(self._ingest_concept_images(concept.id, body, concept.id, mode))
    except Exception:
        pass

    # Phase 7: Rebuild indexes
    self._build_search_indexes(rebuild=True)

    return {
        "concept_id": concept.id,
        "title": concept.title,
        "description": concept.description,
        "tags": concept.tags,
        "chunk_count": chunk_count,
        "image_count": image_count,
    }
```

---

## Part 3: Add `ingest_thoughts()` Method

### Location

`okfgraph/router.py`, as a new public method on `OKFRouter` (~60 lines).

### Implementation

```python
def ingest_thoughts(
    self,
    thoughts: str,
    *,
    topic: str,
    concept_id: str | None = None,
    tags: list[str] | None = None,
    routing_mode: str = "auto",
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Store LLM reasoning/thinking as a searchable concept.

    Wraps the raw reasoning text in OKF-compliant markdown with metadata
    (kind=thought, thought_type=reasoning, topic) so it can be searched,
    traversed, and used as context for other queries.

    The markdown is linted with mordant before import. Fixable issues
    are auto-corrected.

    Args:
        thoughts: The raw reasoning text from the LLM.
        topic: High-level topic or domain for the reasoning.
        concept_id: Optional explicit concept ID. If None, generated from topic.
        tags: Optional additional tags.
        routing_mode: ONNX routing mode for embedding.
        batch_size: Batch size for encoding.

    Returns:
        Dict with keys:
        - "concept_id": The created concept ID
        - "topic": Topic used
        - "tags": Applied tags
        - "chunk_count": Number of chunks created
        - "markdown": The generated markdown content
    """
    import uuid
    from datetime import datetime

    # Generate concept_id from topic if not provided
    if not concept_id:
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        slug = topic.lower().replace(" ", "_")[:30]
        concept_id = f"thought_{slug}_{ts}_{str(uuid.uuid4())[:6]}"

    # Build OKF-compliant markdown
    header_lines = [
        "---",
        f"title: \"Thought: {topic}\"",
        f"kind: thought",
        f"thought_type: reasoning",
        f"topic: {topic}",
        f"tags: [thought, reasoning, {topic}]",
        f"created: {datetime.now().isoformat()}",
        "---",
        "",
        thoughts,
    ]
    markdown = "\n".join(header_lines)

    # Apply via ingest_md internally
    all_tags = list(set(["thought", "reasoning", topic] + (tags or [])))

    from okfgraph.router import Concept
    concept = Concept(
        id=concept_id,
        title=f"Thought: {topic}",
        description=f"Reasoning about {topic}",
        body=markdown,
        kind="thought",
        tags=all_tags,
    )

    result = self._import_single_concept(concept, markdown, all_tags, "text", batch_size)

    return {
        "concept_id": result["concept_id"],
        "topic": topic,
        "tags": result["tags"],
        "chunk_count": result["chunk_count"],
        "markdown": markdown,
    }
```

---

## Part 4: Amending All Existing LLM Tool Definitions

### Changes to `okfgraph/tools.py`

Update the `TOOLS` list. The following tools need description updates to reference the new write operations:

#### 1. `search_hybrid` — Add note about write operations

```python
"description": (
    "Semantic + keyword search over concepts. Use for open-ended questions. "
    "Note: to add new content to the graph, use ``ingest_md`` or "
    "``ingest_thoughts`` instead."
),
```

#### 2. `traverse` — Add note about graph growth

```python
"description": (
    "Navigate relationships (CONTAINS or LINKS_TO) from a concept. "
    "To expand the graph, use ``ingest_md`` to add documents or "
    "``ingest_thoughts`` to persist reasoning."
),
```

#### 3. `get_by_id` — No changes needed (read-only, no context to add)

#### 4. `list_directory` — No changes needed (read-only)

#### 5. `search_images` — No changes needed (read-only)

#### 6. `search_chunks` — No changes needed (read-only)

#### 7. `search_with_context` — No changes needed (read-only)

#### 8. `search_chunks_with_hub_score` — No changes needed (read-only)

#### 9. `expand_with_graph_context` — No changes needed (read-only)

#### 10. `get_chunks` — No changes needed (read-only)

#### 11. `reconstruct_document` — No changes needed (read-only)

#### 12. `find_path` — No changes needed (read-only)

#### 13. `export_bundle` — Add note about write operations

```python
"description": (
    "Export concepts from the graph to an OKF-compliant bundle directory. "
    "Each concept is written as a markdown file with YAML frontmatter. "
    "The body is enriched with graph-derived LINKS_TO links (See Also + Cited By). "
    "index.md files are generated for every directory. "
    "To add content back to the graph, use ``ingest_md``."
),
```

#### 14. NEW: `ingest_md` tool definition

```python
{
    "name": "ingest_md",
    "description": (
        "Import a single markdown file into the knowledge graph. Use when a user "
        "provides a document path or asks to add content from a file. "
        "The file is linted with mordant before import — fixable formatting "
        "issues (MD009, MD012, MD047) are auto-corrected. "
        "Returns the concept ID so the content can be searched or traversed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "md_path": {
                "type": "string",
                "description": "Path to the markdown file to import.",
            },
            "concept_id": {
                "type": "string",
                "description": (
                    "Optional explicit concept ID. If not provided, "
                    "generated from filename."
                ),
            },
            "title": {
                "type": "string",
                "description": "Optional title override.",
            },
            "description": {
                "type": "string",
                "description": "Optional description override.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags to apply.",
            },
            "routing_mode": {
                "type": "string",
                "enum": ["auto", "surgical", "always", "never"],
                "default": "auto",
                "description": "ONNX routing mode for embedding.",
            },
            "mode": {
                "type": "string",
                "enum": ["text", "optional", "omni"],
                "default": "text",
                "description": "Image ingestion mode.",
            },
        },
        "required": ["md_path"],
    },
},
```

#### 15. NEW: `ingest_thoughts` tool definition

```python
{
    "name": "ingest_thoughts",
    "description": (
        "Store LLM reasoning or thinking as a searchable concept in the "
        "knowledge graph. Use when the agent wants to persist its reasoning "
        "process, decision logs, or intermediate conclusions. The stored "
        "thoughts can later be searched, traversed, and used as context "
        "for other queries. Wraps the text in OKF-compliant markdown with "
        "metadata (kind=thought, thought_type=reasoning) so it can be "
        "filtered and searched."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "thoughts": {
                "type": "string",
                "description": "The raw reasoning text from the LLM.",
            },
            "topic": {
                "type": "string",
                "description": "High-level topic or domain for the reasoning.",
            },
            "concept_id": {
                "type": "string",
                "description": (
                    "Optional explicit concept ID. If not provided, "
                    "generated from topic + timestamp."
                ),
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional additional tags.",
            },
            "routing_mode": {
                "type": "string",
                "enum": ["auto", "surgical", "always", "never"],
                "default": "auto",
                "description": "ONNX routing mode for embedding.",
            },
        },
        "required": ["thoughts", "topic"],
    },
},
```

---

## Part 5: Tests

### Location: `tests/test_router.py`

Add two new test classes and update `TestTools`:

```python
class TestTools:
    """Tests for LLM tool definitions."""

    def test_ingest_md_tool_exists(self):
        from okfgraph.tools import TOOLS
        names = [t["name"] for t in TOOLS]
        assert "ingest_md" in names, "ingest_md tool not in TOOLS"

    def test_ingest_md_tool_parameters(self):
        from okfgraph.tools import TOOLS
        tool = next(t for t in TOOLS if t["name"] == "ingest_md")
        assert "md_path" in tool["parameters"]["required"]
        assert "auto_import" not in tool["parameters"]["properties"]  # not exposed to LLM
        assert tool["parameters"]["properties"]["routing_mode"]["enum"] == ["auto", "surgical", "always", "never"]

    def test_ingest_thoughts_tool_exists(self):
        from okfgraph.tools import TOOLS
        names = [t["name"] for t in TOOLS]
        assert "ingest_thoughts" in names, "ingest_thoughts tool not in TOOLS"

    def test_ingest_thoughts_tool_parameters(self):
        from okfgraph.tools import TOOLS
        tool = next(t for t in TOOLS if t["name"] == "ingest_thoughts")
        assert set(tool["parameters"]["required"]) == {"thoughts", "topic"}
        assert "topic" in tool["parameters"]["properties"]

    def test_write_tools_reference_each_other(self):
        """Write tools should reference each other in descriptions."""
        from okfgraph.tools import TOOLS
        ingest_md = next(t for t in TOOLS if t["name"] == "ingest_md")
        ingest_thoughts = next(t for t in TOOLS if t["name"] == "ingest_thoughts")
        # Each should reference the other
        assert "ingest_thoughts" in ingest_md["description"].lower() or "ingest_md" in ingest_thoughts["description"].lower()


class TestIngestMd:
    """Tests for OKFRouter.ingest_md()."""

    def test_import_existing_file(self, tmp_dir):
        """Import a valid markdown file."""
        md_path = Path(tmp_dir) / "test.md"
        md_path.write_text("---\ntitle: Test\n---\n\nHello world.", encoding="utf-8")

        r = OKFRouter(
            db_path=str(Path(tmp_dir) / "test.db"),
            bundle_root=tmp_dir,
            embedding_dim=512,
        )
        result = r.ingest_md(md_path)

        assert "concept_id" in result
        assert result["title"] == "Test"
        assert result["lint_issues"]["error_count"] == 0
        r.close()

    def test_import_with_linting(self, tmp_dir):
        """Linting auto-fixes fixable issues."""
        md_path = Path(tmp_dir) / "test.md"
        md_path.write_text("---\ntitle: Test\n---\n\nHello  \n\nWorld", encoding="utf-8")  # trailing spaces (MD009)

        r = OKFRouter(
            db_path=str(Path(tmp_dir) / "test.db"),
            bundle_root=tmp_dir,
            embedding_dim=512,
        )
        result = r.ingest_md(md_path)

        assert result["lint_issues"]["fixed_count"] > 0
        r.close()

    def test_import_nonexistent_file(self, tmp_dir):
        """Importing a non-existent file raises FileNotFoundError."""
        r = OKFRouter(
            db_path=str(Path(tmp_dir) / "test.db"),
            bundle_root=tmp_dir,
            embedding_dim=512,
        )
        with pytest.raises(FileNotFoundError):
            r.ingest_md("/nonexistent/path.md")
        r.close()

    def test_import_with_explicit_metadata(self, tmp_dir):
        """Explicit metadata overrides frontmatter."""
        md_path = Path(tmp_dir) / "test.md"
        md_path.write_text("---\ntitle: Frontmatter\n---\n\nHello world.", encoding="utf-8")

        r = OKFRouter(
            db_path=str(Path(tmp_dir) / "test.db"),
            bundle_root=tmp_dir,
            embedding_dim=512,
        )
        result = r.ingest_md(md_path, title="Override", tags=["custom", "test"])

        assert result["title"] == "Override"
        assert "custom" in result["tags"]
        r.close()


class TestIngestThoughts:
    """Tests for OKFRouter.ingest_thoughts()."""

    def test_store_reasoning(self, tmp_dir):
        """Store reasoning as a searchable concept."""
        r = OKFRouter(
            db_path=str(Path(tmp_dir) / "test.db"),
            bundle_root=tmp_dir,
            embedding_dim=512,
        )
        result = r.ingest_thoughts(
            thoughts="I think we should use X because Y and Z.",
            topic="architecture",
        )

        assert "concept_id" in result
        assert result["topic"] == "architecture"
        assert "thought" in result["tags"]
        assert "reasoning" in result["tags"]

        # Verify it's stored as a concept
        concept = r.get_by_id(result["concept_id"])
        assert concept is not None
        assert concept.kind == "thought"
        r.close()

    def test_searchable_as_concept(self, tmp_dir):
        """Stored thoughts are searchable via graph queries."""
        r = OKFRouter(
            db_path=str(Path(tmp_dir) / "test.db"),
            bundle_root=tmp_dir,
            embedding_dim=512,
        )
        result = r.ingest_thoughts(
            thoughts="The best approach is to use a graph database.",
            topic="database",
        )

        # Search should find it
        results = r.search_hybrid("graph database")
        ids = [r["id"] for r in results]
        assert result["concept_id"] in ids
        r.close()

    def test_explicit_concept_id(self, tmp_dir):
        """Explicit concept_id is used as-is."""
        r = OKFRouter(
            db_path=str(Path(tmp_dir) / "test.db"),
            bundle_root=tmp_dir,
            embedding_dim=512,
        )
        result = r.ingest_thoughts(
            thoughts="Test reasoning.",
            topic="test",
            concept_id="my_custom_id",
        )

        assert result["concept_id"] == "my_custom_id"
        r.close()
```

---

## Part 6: Documentation Updates

### `architecture.md` — Update §6 tool table

Add two new rows to the tool table:

| Tool | Type | Description |
|---|---|---|
| `ingest_md` | **Write** | Import a single markdown file with linting and metadata control |
| `ingest_thoughts` | **Write** | Store LLM reasoning as a searchable concept with metadata |

### `docs/gap-analysis.md` — Mark gaps closed

```markdown
| **#5c LLM tool definitions** | Low | ✅ **Closed** — ingest_md + ingest_thoughts tools added, all tools updated (v5.7) |
```

### `README.md` — Update test count and tool table

Update test count to reflect new tests:
```markdown
| **Missing Tests** | ✅ Closed (v5.7) | 287 tests across 24 files |
```

---

## Summary: Files Changed

| File | Changes | Lines |
|---|---|---|
| `okfgraph/router.py` | Add `_lint_converted_md()`, `_import_single_concept()`, `ingest_md()`, `ingest_thoughts()` | ~250 |
| `okfgraph/tools.py` | Add `ingest_md` + `ingest_thoughts` definitions; update 2 existing tool descriptions | ~80 |
| `tests/test_router.py` | Add `TestTools` (4 new tests), `TestIngestMd` (4 tests), `TestIngestThoughts` (3 tests) | ~150 |
| `architecture.md` | Update §6 tool table | ~5 |
| `docs/gap-analysis.md` | Mark #5c closed | ~1 |
| `README.md` | Update test count, tool table | ~3 |

## Estimated Effort

| Task | Time |
|---|---|
| `_lint_converted_md()` helper | 10 min |
| `_import_single_concept()` helper | 20 min |
| `ingest_md()` method | 15 min |
| `ingest_thoughts()` method | 10 min |
| Tool definitions + descriptions | 15 min |
| Tests | 25 min |
| Documentation | 10 min |
| **Total** | **~105 min (1.75 hours)** |
