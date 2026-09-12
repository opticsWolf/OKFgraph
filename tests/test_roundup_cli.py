"""CLI coverage for the roundup surface: --rank, --max-tokens, diff,
doctor, --flavor. Help tests are model-free; the workflow class pays one
init+import and reuses it across subprocess calls."""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

FIX = Path(__file__).parent / "fixtures"


def _help(*args):
    result = subprocess.run(
        [sys.executable, "-m", "okfgraph.cli", *args, "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    return result.stdout


class TestRoundupHelp:
    def test_search_help_has_rank(self):
        assert "--rank" in _help("search")

    def test_read_help_has_max_tokens(self):
        assert "--max-tokens" in _help("read")

    def test_export_help_has_flavor(self):
        assert "--flavor" in _help("export")

    def test_diff_help(self):
        out = _help("diff")
        assert "old" in out and "okf --help" in out

    def test_doctor_help(self):
        out = _help("doctor")
        assert "--strict" in out and "--fix" in out


class TestRoundupWorkflow:
    """Live CLI workflow on a scratch graph (one model-loading setup)."""

    @classmethod
    def setup_class(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.db_path = f"{cls.tmpdir}/test.db"
        cls.bundle = f"{cls.tmpdir}/bundle"
        shutil.copytree(FIX / "doctor_bundle", cls.bundle)
        cls._run(["init", "--db", cls.db_path, "--bundle", cls.bundle])
        assert Path(cls.db_path).exists()
        cls._run(["import", "--db", cls.db_path, "--bundle", cls.bundle, "--all"])

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    @classmethod
    def _run(cls, args):
        result = subprocess.run(
            [sys.executable, "-m", "okfgraph.cli"] + args,
            capture_output=True, text=True, timeout=300,
        )
        return result

    def _base(self):
        return ["--db", self.db_path, "--bundle", self.bundle]

    def test_search_rank_ppr(self):
        result = self._run(["search", *self._base(), "--rank", "ppr", "hub spokes"])
        assert result.returncode == 0
        assert "Central Hub" in result.stdout

    def test_search_rank_rejected_for_chunks(self):
        result = self._run(
            ["search", *self._base(), "--target", "chunks", "--rank", "ppr", "hub"])
        assert "concepts-only" in result.stdout

    def test_read_max_tokens(self):
        result = self._run(["read", *self._base(), "hub", "--max-tokens", "40"])
        assert result.returncode == 0
        assert "tokens" in result.stdout

    def test_diff_snapshot_exit_codes(self):
        a, b = FIX / "diff_a", FIX / "diff_b"
        same = self._run(["diff", str(a), str(a)])
        assert same.returncode == 0
        assert "No structural differences" in same.stdout
        changed = self._run(["diff", str(a), str(b)])
        assert changed.returncode == 1
        assert "+ concept gamma" in changed.stdout

    def test_diff_json(self):
        import json
        result = self._run(["diff", str(FIX / "diff_a"), str(FIX / "diff_b"), "--json"])
        assert result.returncode == 1
        report = json.loads(result.stdout)
        assert report["added"] == ["gamma"]

    def test_diff_drift_clean(self):
        result = self._run(["diff", *self._base()])
        assert result.returncode == 0

    def test_doctor_reports_score(self):
        result = self._run(["doctor", *self._base()])
        assert result.returncode == 0
        assert "Health score:" in result.stdout
        assert "broken_link" in result.stdout

    def test_doctor_strict_fails(self):
        result = self._run(["doctor", *self._base(), "--strict"])
        assert result.returncode == 1

    def test_doctor_fix_skips_reviewed(self):
        result = self._run(["doctor", *self._base(), "--fix"])
        assert result.returncode == 0
        assert "dangling" in result.stdout  # skipped reviewed

    def test_export_obsidian_flavor(self):
        out = f"{self.tmpdir}/vault"
        result = self._run(["export", *self._base(), "--all", "--output", out,
                            "--flavor", "obsidian"])
        assert result.returncode == 0
        exported = list(Path(out).rglob("*.md"))
        assert len(exported) >= 6  # every concept, no index files
        assert not list(Path(out).rglob("index.md"))
        # Same graph in okf flavor DOES carry the Cited By section
        # (obsidian omits it: backlinks re-imported as [[..]] would flip).
        out_okf = f"{self.tmpdir}/okfbundle"
        result = self._run(["export", *self._base(), "--all",
                            "--output", out_okf])
        assert result.returncode == 0
        hub = (Path(out_okf) / "hub.md").read_text(encoding="utf-8")
        assert "## Cited By" in hub
        hub_obs = (Path(out) / "hub.md").read_text(encoding="utf-8")
        assert "## Cited By" not in hub_obs
