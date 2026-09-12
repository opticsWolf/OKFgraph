"""Doctor tests: score locked on the fixture bundle, reviewed:true immunity,
and strict/fix behaviour."""

import shutil
import tempfile
from pathlib import Path

import pytest

from okfgraph.components.doctor import is_reviewed, parse_timestamp
from okfgraph.router import OKFRouter

FIX = Path(__file__).parent / "fixtures"


class TestDoctorPure:
    def test_parse_timestamp_variants(self):
        assert parse_timestamp("2020-01-01T00:00:00") is not None
        assert parse_timestamp("05.01.2024") is not None
        assert parse_timestamp("not a date") is None
        assert parse_timestamp(None) is None

    def test_is_reviewed(self):
        assert is_reviewed({"reviewed": "true"})
        assert is_reviewed({"reviewed": True})
        assert not is_reviewed({"reviewed": "false"})
        assert not is_reviewed({})
        assert not is_reviewed(None)


class TestDoctorLive:
    @pytest.fixture(scope="class")
    @classmethod
    def tmp_dir(cls):
        d = tempfile.mkdtemp()
        cls._tmp_dir = d
        yield cls._tmp_dir
        shutil.rmtree(cls._tmp_dir, ignore_errors=True)

    @pytest.fixture(scope="class")
    @classmethod
    def router(cls, tmp_dir):
        r = OKFRouter(
            db_path=str(Path(tmp_dir) / "test_doctor.db"),
            bundle_root=str(FIX / "doctor_bundle"),
            embedding_dim=512,
            chunk_size=50,
            chunk_overlap=10,
            device="cpu",
        )
        ids = r.import_mgr.import_bundle(FIX / "doctor_bundle")
        assert len(ids) == 6, f"expected 6 concepts, got {ids}"
        cls._router = r
        yield cls._router
        cls._router.close()

    def test_score_locked(self, router):
        report = router.diagnose(stale_days=365)
        assert report["concepts"] == 6
        rules = sorted((f["rule"], f["path"]) for f in report["findings"])
        assert ("broken_link", "dangling") in rules
        assert ("orphan", "lonely") in rules
        assert ("orphan", "dangling") in rules  # broken edge is not a real edge
        assert ("duplicate_title", "spoke1") in rules
        assert ("duplicate_title", "spoke2") in rules
        assert ("missing_description", "lonely") in rules
        assert ("stale", "stale") in rules
        # 100 - 5 (broken) - 2*2 (orphans) - 2*3 (dupes) - 1 (nodesc) - 1 (stale)
        assert report["score"] == 83
        assert report["summary"] == {
            "broken_link": 1,
            "duplicate_title": 2,
            "missing_description": 1,
            "orphan": 2,
            "stale": 1,
        }

    def test_hub_info_never_scores(self, router):
        report = router.diagnose()
        assert any(i["rule"] == "hub_concentration" for i in report["info"])
        assert "hub_concentration" not in report["summary"]

    def test_fix_respects_reviewed(self, router):
        fixed = router.doctor_fix()
        assert fixed["skipped_reviewed"] == ["dangling"]
        # The reviewed concept's broken link is still recorded afterwards.
        remaining = { (b["source"], b["target"]) for b in router.list_broken_links() }
        assert ("dangling", "missing-target") in remaining

    def test_findings_sorted(self, router):
        report = router.diagnose()
        keys = [(f["rule"], f["path"]) for f in report["findings"]]
        assert keys == sorted(keys)
