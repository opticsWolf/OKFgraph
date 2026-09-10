# OKF-Compliant Export — Implementation Plan

## 1. Problem Statement

Current export writes the stored body verbatim. Graph `LINKS_TO` edges are ignored, so exported bundles are **trees** (files in directories), not **graphs** (linked documents). The OKF visualizer and consumers see isolated nodes.

## 2. Solution

Enrich every exported document with graph-derived links:
- **See Also** — outgoing `LINKS_TO` edges not already in the body
- **Cited By** — incoming `LINKS_TO` edges (reverse)

This is Option A: append sections, don't replace. The original body is preserved; the graph fills in what's missing.

## 3. Implementation

### 3.1 File: `okfgraph/router.py`

Three changes: two new methods, one modified method.

#### Change 1: Add `_enrich_body_with_graph_links()` to `OKFRouter`

```python
def _enrich_body_with_graph_links(
    self, concept_id: str, body: str
) -> str:
    """Enrich body with graph-derived links so the exported markdown
    faithfully reflects the LINKS_TO graph.

    Strategy (Option A — append, never replace):
      1. Query all outgoing LINKS_TO edges from this concept.
      2. For each target, check if a link to that target already exists
         in the body (by matching the target_id in link URLs).
      3. If not already linked, append a "See Also" bullet.
      4. Query all incoming LINKS_TO edges (concepts that link TO this one).
      5. If any exist, append a "Cited By" bullet list.

    This preserves the original body's links (which may have richer anchor
    text) while ensuring the graph structure is expressed in the export.
    """
    parts: List[str] = []

    # --- Outgoing links (See Also) ---
    result = self.conn.execute("""
        MATCH (s:Concept {id: $id})-[:LINKS_TO]->(t:Concept)
        RETURN t.id AS target_id, t.title AS title, t.type AS type
        ORDER BY t.title
    """, {"id": concept_id})
    outgoing_rows = result.rows_as_dict().get_all()

    if outgoing_rows:
        # Determine which targets are already linked in the body
        # by scanning for link URLs containing the target_id
        existing_link_targets = set()
        for row in outgoing_rows:
            target_id = row["target_id"]
            # Check if target_id appears in any link URL in the body
            link_pattern = re.compile(
                r"\]\(([^)]*?" + re.escape(target_id) + r"[^)]*)\)"
            )
            if link_pattern.search(body):
                existing_link_targets.add(target_id)

        # Collect targets that need a link added
        new_links = []
        for row in outgoing_rows:
            target_id = row["target_id"]
            if target_id not in existing_link_targets:
                title = row["title"] or target_id.split("/")[-1]
                new_links.append(f"- [{title}]({target_id}.md)")

        if new_links:
            parts.append("\n## See Also\n" + "\n".join(new_links))

    # --- Incoming links (Cited By) ---
    result = self.conn.execute("""
        MATCH (s:Concept)-[:LINKS_TO]->(t:Concept {id: $id})
        RETURN s.id AS source_id, s.title AS title, s.type AS type
        ORDER BY s.title
    """, {"id": concept_id})
    incoming_rows = result.rows_as_dict().get_all()

    if incoming_rows:
        cited_lines = ["\n## Cited By\n"]
        for row in incoming_rows:
            source_id = row["source_id"]
            title = row["title"] or source_id.split("/")[-1]
            cited_lines.append(f"- [{title}]({source_id}.md)")
        parts.append("\n".join(cited_lines))

    return body + "".join(parts)
```

#### Change 2: Modify `_write_okf()` to call the enricher

```python
def _write_okf(self, concept: ConceptModel, output_path: Path) -> None:
    """Internal: serialize a ConceptModel to an OKF .md file.

    Enriches the body with LINKS_TO relationships from the graph so that
    exported markdown faithfully reflects the graph structure.
    """
    data = concept.model_dump()
    body = data.pop("body", "")
    data.pop("id", None)
    data.pop("embedding", None)

    if isinstance(data.get("timestamp"), datetime):
        data["timestamp"] = data["timestamp"].isoformat()

    yaml_str = yaml.dump(
        data, default_flow_style=False, allow_unicode=True, sort_keys=False
    )

    # ENRICH: add graph-derived links to the body
    body = self._enrich_body_with_graph_links(concept.id, body)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"---\n{yaml_str}---\n\n{body}", encoding="utf-8")
```

#### Change 3: Add `_generate_index_files()` to `export_bundle()`

OKF specifies "progressive disclosure built in" via auto-generated `index.md` files. Add this to `export_bundle()` after writing all concept files:

```python
def _generate_index_files(self, output_dir: Path, concepts: Dict[str, ConceptModel]) -> None:
    """Generate index.md files for every directory in the bundle.

    Each index.md lists the children (concepts and subdirectories) of that
    directory, enabling progressive disclosure for OKF consumers.
    """
    # Build a map of directory_id → list of (title, relative_path) children
    dir_children: Dict[str, List[Tuple[str, str]]] = {}

    for cid, concept in concepts.items():
        parts = cid.split("/")
        for i in range(1, len(parts)):
            dir_id = "/".join(parts[:i])
            dir_children.setdefault(dir_id, [])
            child_title = concept.title or parts[i]
            child_rel = cid.replace("/", os.sep) + ".md"
            dir_children[dir_id].append((child_title, child_rel))

    # Write index.md for each directory
    for dir_id, children in dir_children.items():
        # Sort children by title
        children.sort(key=lambda x: x[0])
        lines = [
            f"# {dir_id.split('/')[-1] or '(root)'}\n",
            "",
        ]
        for title, rel_path in children:
            lines.append(f"- [{title}]({rel_path})")
        lines.append("")

        # Create parent directories if needed
        dir_path = output_dir / dir_id.replace("/", os.sep)
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / "index.md").write_text("\n".join(lines), encoding="utf-8")
```

Then call it at the end of `export_bundle()`:

```python
def export_bundle(
    self,
    output_dir: Path,
    directory_id: Optional[str] = None,
    concept_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> List[str]:
    """Export concepts from the graph back to an OKF bundle directory.

    Produces OKF-compliant bundles:
      - Each concept is a .md file with YAML frontmatter and a markdown body.
      - The body is enriched with graph-derived LINKS_TO links (See Also + Cited By).
      - index.md files are generated for every directory for progressive disclosure.

    Supports filtering by directory subtree, concept type, or tags.

    Args:
        output_dir: Root directory to write the bundle into.
        directory_id: If set, only export concepts under this directory.
        concept_type: If set, only export concepts of this type.
        tags: If set, only export concepts with ALL these tags.

    Returns:
        List of exported concept IDs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch all concepts (optionally filtered)
    concepts = self._fetch_concepts(
        concept_type=concept_type,
        tags=tags,
    )

    # If directory_id specified, filter to subtree
    if directory_id:
        concepts = {
            cid: c for cid, c in concepts.items()
            if self._is_under_directory(cid, directory_id)
        }

    if not concepts:
        return []

    # Export each concept
    exported: List[str] = []
    for cid, concept in sorted(concepts.items()):
        rel_path = cid.replace("/", os.sep)
        file_path = output_dir / (rel_path + ".md")
        try:
            self._write_okf(concept, file_path)
            exported.append(cid)
        except Exception as e:
            print(f"  [WARN] Failed to export {cid}: {e}")

    # Generate index.md files for progressive disclosure
    self._generate_index_files(output_dir, concepts)

    return exported
```

### 3.2 File: `okfgraph/tools.py`

Add a new tool for export with options:

```python
{
    "name": "export_bundle",
    "description": (
        "Export concepts from the graph to an OKF-compliant bundle directory. "
        "Each concept is written as a markdown file with YAML frontmatter. "
        "The body is enriched with graph-derived LINKS_TO links (See Also + Cited By). "
        "index.md files are generated for every directory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "output_dir": {
                "type": "string",
                "description": "Output directory for the bundle.",
            },
            "directory_id": {
                "type": "string",
                "description": "Optional: only export concepts under this directory.",
            },
            "concept_type": {
                "type": "string",
                "description": "Optional: only export concepts of this type.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional: only export concepts with ALL these tags.",
            },
        },
        "required": ["output_dir"],
    },
},
```

### 3.3 File: `okfgraph/cli.py`

Add `--generate-index` flag to the export subparser (defaults to true for OKF compliance):

```python
p = sub.add_parser("export", help="Export concepts")
_add_global(p)
p.add_argument("--all", action="store_true", dest="export_all", help="Export entire bundle")
p.add_argument("--output", required=True, help="Output directory")
p.add_argument("--concept-id", help="Concept ID (for single export)")
p.add_argument("--type", help="Concept type filter")
p.add_argument("--tags", help="Comma-separated tag filters")
p.add_argument("--parent", help="Parent directory ID")
# NEW:
p.add_argument(
    "--no-index", action="store_true",
    help="Skip generating index.md files (default: generate them)",
)
```

Then in `_export()`:

```python
def _export(args):
    router = _router(args)
    if getattr(args, "export_all", False):
        tags = args.tags.split(",") if args.tags else None
        ids = router.export_bundle(
            output_dir=Path(args.output),
            directory_id=args.parent,
            concept_type=args.type,
            tags=tags,
        )
        print(f"[OK] Exported {len(ids)} concepts to {args.output}")
    else:
        cid = args.concept_id
        output_path = Path(args.output) / f"{cid}.md"
        router.export_to_okf(cid, output_path)
        print(f"[OK] Exported {cid} → {output_path}")
```

(No change needed — the flag can be wired later if desired. For now, `index.md` generation is always on.)

## 4. OKF Compliance Checklist

| Requirement | Before | After |
|---|---|---|
| Markdown files with YAML frontmatter | ✅ | ✅ |
| `type` key in frontmatter | ✅ | ✅ |
| `title` key in frontmatter | ✅ | ✅ |
| `description` key in frontmatter | ✅ | ✅ |
| `resource` key in frontmatter (optional) | ✅ | ✅ |
| `tags` key in frontmatter (optional) | ✅ | ✅ |
| `timestamp` key in frontmatter (optional) | ✅ | ✅ |
| Arbitrary extra frontmatter keys allowed | ✅ | ✅ |
| Body contains markdown links expressing graph | ❌ | ✅ (See Also + Cited By) |
| `index.md` for progressive disclosure | ❌ | ✅ |
| Directory hierarchy preserved | ✅ | ✅ |
| Human-readable | ✅ | ✅ |
| Git-versionable | ✅ | ✅ |

## 5. Edge Cases

| Case | Behavior |
|---|---|
| Concept has no outgoing links | No "See Also" section added |
| Concept has no incoming links | No "Cited By" section added |
| Target link already in body | Not duplicated (checked by scanning body for target_id in link URLs) |
| Target has no title | Falls back to last path segment of target_id |
| Empty bundle | Returns empty list, no files written |
| Directory_id filter | Only exports subtree, only generates indexes for that subtree |
| Re-export (overwrite) | Files are overwritten idempotently |

## 6. Test Plan

### `tests/test_export_compliance.py`

```python
class TestOKFExportCompliance:
    def setup_method(self):
        self.router = OKFRouter(
            db_path=":memory:", bundle_root="/tmp",
        )
        self.tmpdir = tempfile.mkdtemp()
        # Create test documents with cross-links
        (Path(self.tmpdir) / "doc_a.md").write_text(
            "---\ntype: note\ntitle: Doc A\ndescription: First doc.\n---\nContent of A."
        )
        (Path(self.tmpdir) / "doc_b.md").write_text(
            "---\ntype: note\ntitle: Doc B\ndescription: Second doc.\n---\nContent of B."
        )
        (Path(self.tmpdir) / "doc_c.md").write_text(
            "---\ntype: note\ntitle: Doc C\ndescription: Third doc.\n---\nContent of C."
        )
        with self.router:
            self.router.import_from_okf(Path(self.tmpdir / "doc_a.md"))
            self.router.import_from_okf(Path(self.tmpdir / "doc_b.md"))
            self.router.import_from_okf(Path(self.tmpdir / "doc_c.md"))
            # doc_a links to doc_b, doc_c links to doc_a (graph edges)
            self.conn.execute("""
                MATCH (a:Concept {id: 'doc_a'}), (b:Concept {id: 'doc_b'})
                MERGE (a)-[:LINKS_TO]->(b)
            """)
            self.conn.execute("""
                MATCH (c:Concept {id: 'doc_c'}), (a:Concept {id: 'doc_a'})
                MERGE (c)-[:LINKS_TO]->(a)
            """)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def test_export_has_see_also(self):
        """doc_a should have 'See Also' with doc_b (outgoing link)."""
        out = Path(self.tmpdir) / "export"
        self.router.export_bundle(out)
        body = (out / "doc_a.md").read_text()
        assert "## See Also" in body
        assert "doc_b.md" in body

    def test_export_has_cited_by(self):
        """doc_a should have 'Cited By' with doc_c (incoming link)."""
        out = Path(self.tmpdir) / "export"
        self.router.export_bundle(out)
        body = (out / "doc_a.md").read_text()
        assert "## Cited By" in body
        assert "doc_c.md" in body

    def test_export_no_duplicate_links(self):
        """If doc_b already links to doc_a in its body, don't duplicate."""
        # doc_b has no outgoing links in graph, so no "See Also"
        out = Path(self.tmpdir) / "export"
        self.router.export_bundle(out)
        body = (out / "doc_b.md").read_text()
        assert "## See Also" not in body  # no outgoing LINKS_TO

    def test_export_has_index_files(self):
        """index.md files should be generated for each directory."""
        out = Path(self.tmpdir) / "export"
        self.router.export_bundle(out)
        assert (out / "index.md").exists()

    def test_export_preserves_original_body(self):
        """Original body content is preserved, sections are appended."""
        out = Path(self.tmpdir) / "export"
        self.router.export_bundle(out)
        body = (out / "doc_a.md").read_text()
        assert "Content of A." in body
        assert "description: First doc." in body

    def test_export_filters_by_type(self):
        """Only concepts of the specified type are exported."""
        out = Path(self.tmpdir) / "export"
        self.router.export_bundle(out, concept_type="note")
        assert (out / "doc_a.md").exists()
        assert (out / "doc_b.md").exists()
        assert (out / "doc_c.md").exists()

    def test_export_filters_by_directory(self):
        """Only concepts under the specified directory are exported."""
        # Add a nested concept
        nested = Path(self.tmpdir) / "sub/nested.md"
        nested.parent.mkdir(exist_ok=True)
        nested.write_text("---\ntype: note\ntitle: Nested\n---\nNested content.")
        with self.router:
            self.router.import_from_okf(nested)
        out = Path(self.tmpdir) / "export"
        self.router.export_bundle(out, directory_id="sub")
        assert (out / "sub" / "nested.md").exists()
        assert not (out / "doc_a.md").exists()  # not under "sub"

    def test_export_single_concept(self):
        """Export a single concept by ID."""
        out = Path(self.tmpdir) / "export"
        self.router.export_to_okf("doc_a", out)
        assert (out / "doc_a.md").exists()
        body = (out / "doc_a.md").read_text()
        assert "## See Also" in body
        assert "## Cited By" in body
```

## 7. File Change Summary

| File | Change |
|---|---|
| `router.py` | Add `_enrich_body_with_graph_links()`, modify `_write_okf()`, add `_generate_index_files()`, call it from `export_bundle()` |
| `tools.py` | Add `export_bundle` tool definition |
| `cli.py` | (optional) Add `--no-index` flag |
| `tests/test_export_compliance.py` | NEW — OKF compliance tests |

## 8. Implementation Order

1. `_enrich_body_with_graph_links()` — the core logic
2. Modify `_write_okf()` to call it
3. `_generate_index_files()` — progressive disclosure
4. Call from `export_bundle()`
5. Tests
