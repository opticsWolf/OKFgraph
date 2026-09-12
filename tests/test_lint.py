"""okf lint: pre-import bundle gate (no DB, no model).

Fixture bundle (tests/fixtures/lint_bundle/) holds one of each outcome:
clean link pair, unparseable frontmatter, dangling path + wikilink,
ambiguous wikilink (never guessed), typeless file (warnings only), and an
index.md with a stale link that must be invisible (import skips it too).
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import ClassVar

from okfgraph import OKFRouter
from okfgraph.components.lint import lint_bundle

FIXTURE = Path(__file__).parent / "fixtures" / "lint_bundle"


def _by_file(report, kind):
    return {(e["file"], e["rule"]) for e in report[kind]}


class TestLintFixture:
    def test_counts(self):
        report = lint_bundle(FIXTURE)
        assert report["files"] == 8  # index.md not counted
        assert len(report["errors"]) == 4
        assert len(report["warnings"]) == 2
        assert report["clean"] is False

    def test_error_rules(self):
        report = lint_bundle(FIXTURE)
        assert _by_file(report, "errors") == {
            ("bad_parse.md", "parse"),
            ("dangling.md", "dangling_link"),
            ("dangling.md", "dangling_wikilink"),
            ("amb_user.md", "dangling_wikilink"),  # [[Same]] ambiguous → miss
        }

    def test_warning_rules(self):
        report = lint_bundle(FIXTURE)
        assert _by_file(report, "warnings") == {
            ("typeless.md", "missing_type"),
            ("typeless.md", "missing_title"),
        }

    def test_stable_link_messages(self):
        report = lint_bundle(FIXTURE)
        by_file = {e["file"] + e["rule"]: e for e in report["errors"]}
        assert by_file["dangling.mddangling_link"]["target"] == "nope"
        assert by_file["dangling.mddangling_wikilink"]["link"] == "Nobody"

    def test_index_invisible(self):
        report = lint_bundle(FIXTURE)
        assert all(e["file"] != "index.md" for e in report["errors"])
        assert all(w["file"] != "index.md" for w in report["warnings"])

    def test_sorted_deterministic(self):
        again = lint_bundle(FIXTURE)
        assert lint_bundle(FIXTURE)["errors"] == again["errors"]
        assert lint_bundle(FIXTURE)["warnings"] == again["warnings"]

    def test_clean_bundle(self, tmp_path):
        (tmp_path / "a.md").write_text(
            "---\ntype: note\ntitle: A\n---\nSee [B](b.md).\n")
        (tmp_path / "b.md").write_text("---\ntype: note\ntitle: B\n---\nHi.\n")
        report = lint_bundle(tmp_path)
        assert report["clean"] is True and report["files"] == 2


class TestLintCli:
    """CLI exit codes via subprocess (no router build, no model load)."""

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "okfgraph.cli", "lint", *args],
            capture_output=True, text=True, cwd=Path(__file__).parent.parent,
        )

    def test_dirty_exit_1(self):
        proc = self._run(str(FIXTURE))
        assert proc.returncode == 1
        assert "4 error(s)" in proc.stdout

    def test_json_exit_1(self):
        proc = self._run(str(FIXTURE), "--json")
        assert proc.returncode == 1
        report = json.loads(proc.stdout)
        assert report["files"] == 8 and report["clean"] is False

    def test_clean_exit_0(self, tmp_path):
        (tmp_path / "a.md").write_text("---\ntype: note\ntitle: A\n---\nHi.\n")
        proc = self._run(str(tmp_path))
        assert proc.returncode == 0
        assert "lint-clean" in proc.stdout

    def test_missing_dir_exit_2(self, tmp_path):
        proc = self._run(str(tmp_path / "nope"))
        assert proc.returncode == 2


class TestLintDoctorAgreement:
    """The decision-record lock: lint-clean → import → zero broken_link."""

    router: ClassVar[OKFRouter]
    tmp_dir: ClassVar[str]

    @classmethod
    def setup_class(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        cls.router = OKFRouter(
            db_path=os.path.join(cls.tmp_dir, "agree.db"),
            bundle_root=cls.tmp_dir,
            device="cuda",
        )
        cls.router.__enter__()

    @classmethod
    def teardown_class(cls):
        try:
            cls.router.close()
        finally:
            shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_clean_import_has_no_broken_links(self):
        bundle = Path(self.tmp_dir) / "agree"
        bundle.mkdir(parents=True, exist_ok=True)
        shutil.copy(FIXTURE / "good.md", bundle / "good.md")
        shutil.copy(FIXTURE / "other.md", bundle / "other.md")
        assert lint_bundle(bundle)["clean"] is True
        self.router.import_mgr.import_bundle(bundle)
        report = self.router.diagnose()
        assert [f for f in report["findings"] if f["rule"] == "broken_link"] == []
