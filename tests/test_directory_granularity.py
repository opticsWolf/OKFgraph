"""WS3 step 1: a directory's hash covers its direct children only.

A nested edit queues only its own folder, not every ancestor (R29-G,
amplification 2). The import reports queued vs actually-changed counts (R29-G
remedy 3). Without ``--prune-missing``, a vanished source file is reported
once, by name (R29-G').
"""

import logging
import shutil
import tempfile
from pathlib import Path

import pytest

from okfgraph.components.import_ import is_concept_file
from okfgraph.router import OKFRouter


def _write(root: Path, rel: str, body: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\ntitle: {path.stem}\n---\n{body}\n", encoding="utf-8")


def _sources(root: Path):
    return sorted(p for p in root.rglob("*") if p.is_file() and is_concept_file(p))


@pytest.fixture
def bundle():
    tmp = Path(tempfile.mkdtemp()).resolve()
    root = tmp / "kb"
    _write(root, "top.md", "top level note")
    _write(root, "architecture/x.md", "architecture note")
    _write(root, "architecture/sub/y.md", "nested note")
    _write(root, "architecture/sub/z.md", "another nested note")
    router = OKFRouter(db_path=str(tmp / "kb.db"), bundle_root=str(root),
                       embedding_dim=512, chunk_size=50, chunk_overlap=10,
                       device="cpu")
    router.import_bundle()
    try:
        yield {"router": router, "root": root}
    finally:
        router.close()
        shutil.rmtree(tmp, ignore_errors=True)


def _changed(router, root):
    changed, _deleted, _dirs = router.delta_mgr._changed_directories(
        _sources(root), walk_root=root)
    return sorted(p.relative_to(root).as_posix() for p in changed)


def test_unchanged_bundle_queues_nothing(bundle):
    assert _changed(bundle["router"], bundle["root"]) == []


def test_nested_edit_queues_only_its_own_directory(bundle):
    root = bundle["root"]
    _write(root, "architecture/sub/y.md", "nested note, edited")
    assert _changed(bundle["router"], root) == [
        "architecture/sub/y.md", "architecture/sub/z.md"]


def test_root_edit_does_not_queue_subfolders(bundle):
    root = bundle["root"]
    _write(root, "top.md", "top level note, edited")
    assert _changed(bundle["router"], root) == ["top.md"]


def test_delta_counts_queued_versus_actually_changed(bundle):
    router, root = bundle["router"], bundle["root"]
    _write(root, "architecture/sub/y.md", "nested note, edited")
    report = router.import_bundle()["delta"]
    assert report == {"changed_dirs": 1, "queued": 2, "actually_changed": 1}


def test_unchanged_import_reports_zero(bundle):
    assert bundle["router"].import_bundle()["delta"] == {
        "changed_dirs": 0, "queued": 0, "actually_changed": 0}


def test_vanished_source_is_reported_once_without_prune(bundle, caplog):
    router, root = bundle["router"], bundle["root"]
    (root / "architecture" / "x.md").unlink()
    with caplog.at_level(logging.WARNING, logger="okfgraph.components.import_"):
        router.import_bundle()
    messages = [r.getMessage() for r in caplog.records
                if "no source file anymore" in r.getMessage()]
    assert len(messages) == 1
    assert "architecture/x" in messages[0]
    assert "--prune-missing" in messages[0]
