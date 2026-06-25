"""Integration test: Jina v5 model download + real LadybugDB + OKFRouter."""

import json
import sys
import tempfile
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import yaml

# ------------------------------------------------------------------
# 1. Test Jina v5 model download
# ------------------------------------------------------------------

print("=" * 60)
print("STEP 1: Downloading Jina v5 model...")
print("=" * 60)

from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

MODEL_ID = "jinaai/jina-embeddings-v5-text-small-retrieval"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
print(f"[OK] Tokenizer loaded: {type(tokenizer).__name__}")

embedder = ORTModelForFeatureExtraction.from_pretrained(
    MODEL_ID,
    export=False,
    subfolder="onnx",
)
print(f"[OK] ONNX model loaded: {type(embedder).__name__}")

# Test encoding
import numpy as np
import torch

def encode(text: str, task: str = "Document") -> list[float]:
    """Encode text with prefix, mean pooling, and L2 normalization."""
    if not text.startswith(("Query:", "Document:")):
        text = f"{task}: {text}"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=8192,
        padding=True,
    )
    outputs = embedder(**inputs)

    attention_mask = inputs["attention_mask"]
    mask_expanded = attention_mask.unsqueeze(-1).expand(
        outputs.last_hidden_state.size()
    ).float()
    sum_embeddings = (outputs.last_hidden_state * mask_expanded).sum(dim=1)
    sum_mask = mask_expanded.sum(dim=1)
    pooled = sum_embeddings / sum_mask
    normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)

    return normalized.squeeze().tolist()


# Test document encoding
doc_embedding = encode("Hello, this is a test document about Python programming.", task="Document")
print(f"[OK] Document encoding: {len(doc_embedding)} dimensions (expected 1024 full model)")
assert len(doc_embedding) == 1024, f"Expected 1024 dims from full model, got {len(doc_embedding)}"

# Test query encoding
query_embedding = encode("Python programming tutorial", task="Query")
print(f"[OK] Query encoding: {len(query_embedding)} dimensions (expected 1024 full model)")
assert len(query_embedding) == 1024, f"Expected 1024 dims from full model, got {len(query_embedding)}"

# Test Matryoshka truncation
for target_dim in (384, 256, 512):
    truncated = doc_embedding[:target_dim]
    # Re-normalize truncated vector
    truncated_np = np.array(truncated)
    truncated_np = truncated_np / np.linalg.norm(truncated_np)
    query_np = np.array(query_embedding[:target_dim])
    query_np = query_np / np.linalg.norm(query_np)
    sim_trunc = np.dot(truncated_np, query_np)
    print(f"[OK] Matryoshka {target_dim}d: cosine sim = {sim_trunc:.4f}")

# Test cosine similarity (should be high for related texts)
doc_np = np.array(doc_embedding)
query_np = np.array(query_embedding)
cosine_sim = np.dot(doc_np, query_np)
print(f"[OK] Cosine similarity (related texts, 1024d): {cosine_sim:.4f}")

# Test with unrelated texts
unrelated_embedding = encode("Quantum physics and particle accelerators", task="Document")
unrelated_np = np.array(unrelated_embedding)
cosine_unrelated = np.dot(doc_np, unrelated_np)
print(f"[OK] Cosine similarity (unrelated texts, 1024d): {cosine_unrelated:.4f} (expected < related)")

print("\n" + "=" * 60)
print("STEP 1 PASSED: Jina v5 model works correctly!")
print("=" * 60 + "\n")

# ------------------------------------------------------------------
# 2. Test LadybugDB with OKFRouter
# ------------------------------------------------------------------

print("=" * 60)
print("STEP 2: Creating LadybugDB + OKFRouter...")
print("=" * 60)

# Create temp directory for DB and bundle
with tempfile.TemporaryDirectory() as tmpdir:
    db_path = str(Path(tmpdir) / "okfgraph.db")
    bundle_root = Path(tmpdir) / "bundle"
    bundle_root.mkdir()

    # Create sample OKF files
    sample_files = {
        "introduction.md": {
            "type": "chapter",
            "title": "Introduction to OKF",
            "description": "Overview of the OKF knowledge format",
            "tags": ["okf", "intro"],
        },
        "concepts/basics.md": {
            "type": "section",
            "title": "Basic Concepts",
            "description": "Fundamental concepts you need to know",
            "tags": ["basics", "concepts"],
        },
        "concepts/advanced.md": {
            "type": "section",
            "title": "Advanced Topics",
            "description": "Deep dive into advanced features",
            "tags": ["advanced", "concepts"],
        },
        "tables/data_types.md": {
            "type": "table",
            "title": "Data Types Reference",
            "description": "Complete reference for all data types",
            "tags": ["reference", "types"],
        },
    }

    for filename, frontmatter in sample_files.items():
        filepath = bundle_root / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        yaml_str = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
        filepath.write_text(
            f"---\n{yaml_str}---\n\n# {frontmatter['title']}\n\nThis is sample content about {frontmatter['title'].lower()}.\n",
            encoding="utf-8",
        )

    # Import OKFRouter
    from okfgraph.router import OKFRouter

    print(f"DB path: {db_path}")
    print(f"Bundle root: {bundle_root}")

    # Create router (this will download model + create schema)
    print("\nInitializing OKFRouter (downloads model + creates schema)...")
    router = OKFRouter(
        db_path=db_path,
        bundle_root=str(bundle_root),
        model_id=MODEL_ID,
        embedding_dim=384,  # Matryoshka truncation: 384 of 1024
    )
    print(f"[OK] OKFRouter initialized (embedding_dim={router.embedding_dim})")

    # Import all sample files
    print("\n" + "-" * 40)
    print("Importing OKF files...")
    print("-" * 40)
    imported_ids = []
    for filename in sample_files:
        filepath = bundle_root / filename
        concept_id = router.import_from_okf(filepath)
        imported_ids.append(concept_id)
        print(f"  [OK] Imported: {filename} -> {concept_id}")

    print(f"\n[OK] Total imported: {len(imported_ids)} concepts")

    # Benchmark: batch import vs single-file import
    print("\n" + "-" * 40)
    print("Benchmark: Batch vs Single-File Import")
    print("-" * 40)

    # Create 10 additional files for benchmarking
    bench_dir = bundle_root / "bench"
    bench_dir.mkdir()
    for i in range(10):
        fp = bench_dir / f"doc_{i}.md"
        fm = {"type": "note", "title": f"Benchmark Doc {i}", "tags": ["bench"]}
        yaml_str = yaml.dump(fm, default_flow_style=False, sort_keys=False)
        fp.write_text(
            f"---\n{yaml_str}---\n\n# Benchmark Doc {i}\n\nContent for document number {i}. This is extra text to make the embedding more meaningful for batch processing comparison.\n",
            encoding="utf-8",
        )

    import time

    # Single-file import timing
    t0 = time.perf_counter()
    single_ids = []
    for i in range(10):
        cid = router.import_from_okf(bench_dir / f"doc_{i}.md")
        single_ids.append(cid)
    t_single = time.perf_counter() - t0
    print(f"  [OK] Single-file import (10 files): {t_single:.3f}s ({t_single*100:.1f} ms/file)")

    # Batch import timing
    t0 = time.perf_counter()
    batch_ids = router.import_bundle(bench_dir, batch_size=10)
    t_batch = time.perf_counter() - t0
    print(f"  [OK] Batch import (10 files):     {t_batch:.3f}s ({t_batch*100:.1f} ms/file)")

    speedup = t_single / t_batch if t_batch > 0 else float('inf')
    print(f"  [OK] Speedup: {speedup:.1f}x (batch vs single)")

    # Test get_by_id
    print("\n" + "-" * 40)
    print("Testing get_by_id...")
    print("-" * 40)
    for cid, expected_title, expected_type in [
        ("introduction", "Introduction to OKF", "chapter"),
        ("concepts/basics", "Basic Concepts", "section"),
        ("concepts/advanced", "Advanced Topics", "section"),
        ("tables/data_types", "Data Types Reference", "table"),
    ]:
        concept = router.get_by_id(cid)
        assert concept is not None, f"Concept {cid} not found"
        assert concept.title == expected_title
        print(f"  [OK] {cid}: {concept.title} ({concept.type})")

    # Test list_directory
    print("\n" + "-" * 40)
    print("Testing list_directory (root)...")
    print("-" * 40)
    root_items = router.list_directory("")
    for item in root_items:
        print(f"  {item['type']:10s} {item['title']}")

    print("\nTesting list_directory (concepts/)...")
    concepts_items = router.list_directory("concepts")
    for item in concepts_items:
        print(f"  {item['type']:10s} {item['title']}")

    # Test traverse
    print("\n" + "-" * 40)
    print("Testing traverse (CONTAINS from 'concepts')...")
    print("-" * 40)
    traversal = router.traverse(
        start_id="concepts",
        relationship="CONTAINS",
        direction="OUTGOING",
        depth=1,
    )
    for item in traversal:
        print(f"  [OK] {item['id']}: {item['title']}")

    # Test hybrid search
    print("\n" + "-" * 40)
    print("Testing hybrid search: 'advanced concepts'...")
    print("-" * 40)
    results = router.search_hybrid("advanced concepts", limit=5)
    for i, result in enumerate(results, 1):
        print(f"  {i}. [{result['relevance_score']:.4f}] {result['title']} ({result['type']})")
        desc = result['description'] or ''
        print(f"     {desc[:100]}...")

    print("\nTesting hybrid search: 'data types reference'...")
    results = router.search_hybrid("data types reference", limit=3)
    for i, result in enumerate(results, 1):
        print(f"  {i}. [{result['relevance_score']:.4f}] {result['title']} ({result['type']})")

    # Test export
    print("\n" + "-" * 40)
    print("Testing export_to_okf...")
    print("-" * 40)
    export_path = Path(tmpdir) / "exported.md"
    router.export_to_okf(imported_ids[0], export_path)
    exported_content = export_path.read_text(encoding="utf-8")
    print(f"  [OK] Exported to {export_path}")
    print(f"  Content preview:\n{exported_content[:200]}...")

    # Verify exported content is valid OKF
    import frontmatter
    post = frontmatter.loads(exported_content)
    print(f"  [OK] Re-parsed frontmatter: {dict(post.metadata)}")

    # Test bulk export
    print("\n" + "-" * 40)
    print("Testing export_bundle (full)...")
    print("-" * 40)
    export_root = Path(tmpdir) / "exported_bundle"
    exported_ids = router.export_bundle(export_root)
    print(f"  [OK] Exported {len(exported_ids)} concepts to {export_root}")

    # Verify file structure
    exported_files = sorted(export_root.rglob("*.md"))
    print(f"  [OK] Files on disk: {len(exported_files)}")
    for fp in exported_files[:6]:
        rel = fp.relative_to(export_root)
        print(f"       {rel}")
    if len(exported_files) > 6:
        print(f"       ... and {len(exported_files) - 6} more")

    # Verify each exported file is valid OKF
    for fp in exported_files:
        content = fp.read_text(encoding="utf-8")
        post = frontmatter.loads(content)
        assert post.metadata.get("type") is not None, f"Missing type in {fp}"
    print(f"  [OK] All {len(exported_files)} exported files are valid OKF")

    # Test filtered export (by type)
    print("\n" + "-" * 40)
    print("Testing export_bundle (filtered: type=chapter)...")
    print("-" * 40)
    chapter_export = Path(tmpdir) / "chapter_only"
    chapter_ids = router.export_bundle(chapter_export, concept_type="chapter")
    print(f"  [OK] Exported {len(chapter_ids)} chapters")
    assert len(chapter_ids) == 1, f"Expected 1 chapter, got {len(chapter_ids)}"
    assert chapter_ids[0] == "introduction", f"Expected 'introduction', got {chapter_ids[0]}"

    # Test filtered export (by directory)
    print("\n" + "-" * 40)
    print("Testing export_bundle (filtered: directory=concepts)...")
    print("-" * 40)
    dir_export = Path(tmpdir) / "concepts_only"
    dir_ids = router.export_bundle(dir_export, directory_id="concepts")
    print(f"  [OK] Exported {len(dir_ids)} concepts from 'concepts/' directory")
    assert len(dir_ids) == 2, f"Expected 2 concepts in 'concepts/', got {len(dir_ids)}"
    for cid in sorted(dir_ids):
        print(f"       {cid}")

    # Test filtered export (by tags)
    print("\n" + "-" * 40)
    print("Testing export_bundle (filtered: tags=['okf', 'intro'])...")
    print("-" * 40)
    tag_export = Path(tmpdir) / "tagged_only"
    tag_ids = router.export_bundle(tag_export, tags=["okf", "intro"])
    print(f"  [OK] Exported {len(tag_ids)} concepts with tags ['okf', 'intro']")
    assert len(tag_ids) == 1, f"Expected 1 tagged concept, got {len(tag_ids)}"

# ------------------------------------------------------------------
# Broken Links
# ------------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP: Broken Links Tracking & Repair")
print("=" * 60)

# Create a new temp directory for this test
import tempfile
broken_tmp = tempfile.TemporaryDirectory()
tmpdir_broken = Path(broken_tmp.name)
test_bundle = tmpdir_broken / "broken_links_test"
test_bundle.mkdir(exist_ok=True)

# Create a file with a link to a non-existent concept
broken_file = test_bundle / "with_broken_link.md"
yaml_str = yaml.dump({"type": "note", "title": "Has Broken Link"}, default_flow_style=False)
broken_file.write_text(f"---\n{yaml_str}---\n\nThis links to [missing concept](missing.md)\n")

# Import the file — the broken link should be tracked
router.bundle_root = test_bundle  # Update bundle root for this test
cid = router.import_from_okf(broken_file)
print(f"  [OK] Imported {cid}")

# Check for broken links
broken = router.list_broken_links()
print(f"  [OK] Found {len(broken)} broken link(s)")
assert len(broken) == 1, f"Expected 1 broken link, got {len(broken)}"
assert broken[0]["source"] == cid
assert broken[0]["target"] == "missing"

# Now create the missing concept and import it
missing_file = test_bundle / "missing.md"
yaml_str = yaml.dump({"type": "note", "title": "Missing Concept"}, default_flow_style=False)
missing_file.write_text(f"---\n{yaml_str}---\n\nI was missing, now I exist.\n")
missing_cid = router.import_from_okf(missing_file)
print(f"  [OK] Imported missing concept: {missing_cid}")

# Broken link should still be tracked (not auto-repaired on import)
broken_after = router.list_broken_links()
assert len(broken_after) == 1, f"Expected 1 broken link before repair, got {len(broken_after)}"

# Repair links
repaired = router.repair_links()
print(f"  [OK] Repaired {repaired} link(s)")
assert repaired == 1, f"Expected 1 repaired link, got {repaired}"

# Verify no more broken links
broken_final = router.list_broken_links()
assert len(broken_final) == 0, f"Expected 0 broken links after repair, got {len(broken_final)}"

# Verify the LINKS_TO relationship exists
traversal = router.traverse(cid, relationship="LINKS_TO", direction="OUTGOING", depth=1)
assert len(traversal) == 1, f"Expected 1 LINKS_TO target, got {len(traversal)}"
assert traversal[0]["id"] == "missing"
print(f"  [OK] LINKS_TO relationship verified: {cid} → missing")

# ------------------------------------------------------------------
# exclude_reserved
# ------------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP: exclude_reserved Filter")
print("=" * 60)

# Create index and log files that should be excluded
index_file = test_bundle / "index.md"
yaml_str = yaml.dump({"type": "index", "title": "Index File"}, default_flow_style=False)
index_file.write_text(f"---\n{yaml_str}---\n\nThis is an index file.\n")

log_file = test_bundle / "log.md"
yaml_str = yaml.dump({"type": "log", "title": "Log File"}, default_flow_style=False)
log_file.write_text(f"---\n{yaml_str}---\n\nThis is a log file.\n")

router.import_from_okf(index_file)
router.import_from_okf(log_file)
print(f"  [OK] Imported index.md and log.md")

# Search with exclude_reserved=True (default) should NOT include index/log
results_excluded = router.search_hybrid("file", exclude_reserved=True)
result_ids = [r["id"] for r in results_excluded]
print(f"  [OK] Search with exclude_reserved=True: {len(results_excluded)} results")
assert "index" not in result_ids, "index.md should be excluded"
assert "log" not in result_ids, "log.md should be excluded"

# Search with exclude_reserved=False should include index/log
results_included = router.search_hybrid("file", exclude_reserved=False)
result_ids_included = [r["id"] for r in results_included]
print(f"  [OK] Search with exclude_reserved=False: {len(results_included)} results")
assert len(results_included) > len(results_excluded), "exclude_reserved=False should return more results"

print("\n" + "=" * 60)
print("ALL INTEGRATION TESTS PASSED!")
print("=" * 60)

# Cleanup
broken_tmp.cleanup()
