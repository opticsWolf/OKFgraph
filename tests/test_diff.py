"""Structural diff tests: snapshot mode locks the report, drift mode proves
graph-vs-bundle comparison. Uses the fixture pairs diff_a/diff_b."""

import json
from pathlib import Path

import pytest

from okfgraph.components.diff import DiffManager, content_hash, state_of_dir
from okfgraph.router import OKFRouter

FIX = Path(__file__).parent / "fixtures"


class TestDiffPure:
    def test_snapshot_report_locked(self):
        report = DiffManager(None).diff_dirs(FIX / "diff_a", FIX / "diff_b")
        assert report["added"] == ["gamma"]
        assert report["removed"] == []
        assert report["changed"] == ["beta"]
        assert report["retitled"] == [
            {"id": "beta", "old": "Beta Concept", "new": "Beta Concept Renamed"}
        ]
        assert report["retyped"] == []
        assert report["edges_added"] == [["beta", "gamma"]]
        assert report["edges_removed"] == []  # alpha->beta survives (alpha.md copied)
        assert report["broken_new"] == []
        assert report["broken_fixed"] == []
        assert report["identical"] is False

    def test_identical_dir_diffs_clean(self):
        report = DiffManager(None).diff_dirs(FIX / "diff_a", FIX / "diff_a")
        assert report["identical"] is True
        assert report["added"] == report["removed"] == report["changed"] == []

    def test_broken_links_detected(self, tmp_path):
        a = tmp_path / "a"
        a.mkdir()
        (a / "x.md").write_text(
            "---\ntitle: X\ntype: note\n---\n\nSee [Ghost](ghost.md).\n"
        )
        report = DiffManager(None).diff_dirs(a, a)
        assert report["broken_new"] == []
        state = state_of_dir(a)
        assert ("x", "ghost") in state.broken

    def test_content_hash_normalizes(self):
        assert content_hash("a\r\nb  ") == content_hash("a\nb")


class TestDiffDrift:
    """Drift mode on a live graph: import A, diff against A (clean) and B."""

    @pytest.fixture(scope="class")
    @classmethod
    def tmp_dir(cls):
        import shutil
        import tempfile
        d = tempfile.mkdtemp()
        cls._tmp_dir = d
        yield cls._tmp_dir
        shutil.rmtree(cls._tmp_dir, ignore_errors=True)

    @pytest.fixture(scope="class")
    @classmethod
    def router(cls, tmp_dir):
        r = OKFRouter(
            db_path=str(Path(tmp_dir) / "test_diff.db"),
            bundle_root=str(FIX / "diff_a"),
            embedding_dim=512,
            chunk_size=50,
            chunk_overlap=10,
            device="cpu",
        )
        ids = r.import_mgr.import_bundle(FIX / "diff_a")
        assert len(ids) == 2
        cls._router = r
        yield cls._router
        cls._router.close()

    def test_no_drift_after_import(self, router):
        report = router.diff_db_dir(FIX / "diff_a")
        assert report == json.loads(json.dumps(report))  # JSON-serializable
        assert report["identical"] is True

    def test_drift_detects_bundle_changes(self, router):
        report = router.diff_db_dir(FIX / "diff_b")
        assert report["identical"] is False
        assert report["added"] == ["gamma"]
        assert report["changed"] == ["beta"]
