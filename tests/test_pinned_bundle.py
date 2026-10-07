"""Pinned bundle-path + lint diagnostics (0.10.1 Issue B).

B1: relative ``--bundle-path`` crashed on 0.10.0 (mixed relative/absolute
anchors in ``delta.py:158``), and absolute trees outside the configured
root crashed the same way. Pinned trees now resolve once at the boundary
and import isolated: every file imports, nothing tombstones, and nothing
is written to the shared DirHash/FileHash ledger (no collision with the
owned tree's baseline).

B2: ``lint`` of a dot-dir reported ``0 file(s)`` plus an active
``lint-clean (safe to import)`` endorsement of a tree it never scanned.
It now reports the shared skipped count and withholds the endorsement
when the walk was fully filtered.
"""

from pathlib import Path

from okfgraph.components.lint import lint_bundle
from okfgraph.router import OKFRouter


def _write_okf(path: Path, title: str, body: str = "Body text here.") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\ntype: note\ntitle: {title}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return path


def _router(db: Path, root: Path) -> OKFRouter:
    return OKFRouter(db_path=str(db), bundle_root=str(root), device="cpu")


class TestPinnedBundlePath:
    def test_relative_inside(self, tmp_path, monkeypatch):
        """Incident repro: relative --bundle-path inside the root imports."""
        repo = tmp_path / "repo"
        sub = repo / "merge-tmp" / "thoughts" / "topic"
        _write_okf(sub / "rel.md", "Rel")
        r = _router(tmp_path / "t.db", repo)
        try:
            monkeypatch.chdir(repo)
            ids = r.import_mgr.import_bundle(bundle_path="merge-tmp")
        finally:
            r.close()
        assert ids == ["thoughts/topic/rel"]

    def test_absolute_outside(self, tmp_path):
        """New crash class (Amendment A §8.3): outside-root absolute works."""
        repo = tmp_path / "repo"
        repo.mkdir()
        bundle = tmp_path / "ext" / "bundle"
        _write_okf(bundle / "thoughts" / "topic" / "p.md", "P")
        r = _router(tmp_path / "t.db", repo)
        try:
            ids = r.import_mgr.import_bundle(bundle_path=bundle)
        finally:
            r.close()
        assert ids == ["thoughts/topic/p"]

    def test_forward_slash_form(self, tmp_path):
        """Forward-slash absolute form (the report's working variant) works."""
        repo = tmp_path / "repo"
        repo.mkdir()
        bundle = tmp_path / "ext"
        _write_okf(bundle / "thoughts" / "topic" / "s.md", "S")
        r = _router(tmp_path / "t.db", repo)
        try:
            ids = r.import_mgr.import_bundle(
                bundle_path=str(bundle).replace("\\", "/"))
        finally:
            r.close()
        assert ids == ["thoughts/topic/s"]

    def test_repeated_import_stable_and_ledger_clean(self, tmp_path):
        """Same IDs twice; the owned ledger is never polluted/collided."""
        repo = tmp_path / "repo"
        repo.mkdir()
        bundle = tmp_path / "ext"
        _write_okf(bundle / "thoughts" / "topic" / "p.md", "P")
        r = _router(tmp_path / "t.db", repo)
        try:
            first = r.import_mgr.import_bundle(bundle_path=bundle)
            second = r.import_mgr.import_bundle(bundle_path=bundle)
            dir_rows = r.conn.execute(
                "MATCH (d:DirHash) RETURN d.path AS p"
            ).rows_as_dict().get_all()
            file_rows = r.conn.execute(
                "MATCH (f:FileHash) RETURN f.path AS p"
            ).rows_as_dict().get_all()
        finally:
            r.close()
        assert first == ["thoughts/topic/p"]
        assert second == ["thoughts/topic/p"]  # stable, no crash
        assert (dir_rows or []) == []  # isolated: no ledger writes
        assert (file_rows or []) == []  # isolated: no ledger writes


class TestLintSkippedDirs:
    def test_dot_dir_reports_skipped(self, tmp_path):
        dot = tmp_path / ".merge-tmp" / "thoughts" / "topic"
        _write_okf(dot / "hidden.md", "Hidden")
        report = lint_bundle(tmp_path / ".merge-tmp")
        assert report["files"] == 0
        assert report["skipped_hidden"] == 1
        assert report["clean"] is True  # no errors — but see render guard

    def test_render_withholds_safe_claim(self, tmp_path, capsys):
        from okfgraph.cli import _render_lint

        dot = tmp_path / ".merge-tmp" / "thoughts" / "topic"
        _write_okf(dot / "hidden.md", "Hidden")
        _render_lint(lint_bundle(tmp_path / ".merge-tmp"))
        out = capsys.readouterr().out
        assert "skipped 1 file(s) in hidden/tool dirs" in out
        assert "safe to import" not in out

    def test_clean_bundle_still_endorsed(self, tmp_path, capsys):
        from okfgraph.cli import _render_lint

        _write_okf(tmp_path / "a.md", "A")
        _render_lint(lint_bundle(tmp_path))
        out = capsys.readouterr().out
        assert "safe to import" in out


class TestLockAndPurge:
    """Phase 3 hardening: stale-lock info + immediate-purge path."""

    def test_doctor_reports_stale_lock(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        r = _router(tmp_path / "t.db", repo)
        try:
            (tmp_path / "t.db.lock").touch()
            report = r.doctor()["report"]
        finally:
            r.close()
        assert any(i["rule"] == "stale_lock" for i in report["info"])
        assert report["score"] <= 100  # info-only: never a finding

    def test_purge_older_than_zero(self, tmp_path):
        """``--older-than 0`` is the documented force path (no alias)."""
        repo = tmp_path / "repo"
        doc = _write_okf(repo / "doomed.md", "Doomed")
        r = _router(tmp_path / "t.db", repo)
        try:
            cid = r.import_file(doc)["concept_id"]
            r.delete(cid)
            assert r.purge_deleted(older_than=0)["purged"] == 1
        finally:
            r.close()
