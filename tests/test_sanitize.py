"""resource: URI sanitization — credentials never persist into the graph.

Unit matrix for sanitize_resource() plus one live round trip:
import with a credentialed resource: → stored sanitized → re-exported
sanitized.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import ClassVar

import pytest

from okfgraph import OKFRouter
from okfgraph.components.import_ import parse_source_file, sanitize_resource


class TestSanitizeResource:
    @pytest.mark.parametrize("raw,expected", [
        # Credentialed URIs are rewritten ...
        ("mysql://u:p@host:3306/db", "mysql://***@host:3306/db"),
        ("mysql+pymysql://user:pass@host/db?ssl-mode=REQUIRED",
         "mysql+pymysql://***@host/db?ssl-mode=REQUIRED"),
        ("mongodb+srv://u:p@cluster.net/mydb", "mongodb+srv://***@cluster.net/mydb"),
        ("postgres://u@host/db", "postgres://***@host/db"),  # user, no password
        # ... everything else passes through untouched.
        ("https://example.com/docs/a.md", "https://example.com/docs/a.md"),
        ("docs/meeting@noon.md", "docs/meeting@noon.md"),  # @ outside authority
        ("/abs/path/file.md", "/abs/path/file.md"),
        ("rel/path/file.md", "rel/path/file.md"),
        ("mailto:someone@example.com", "mailto:someone@example.com"),
        ("okf-asset://550e8400-e29b-41d4-a716-446655440000",
         "okf-asset://550e8400-e29b-41d4-a716-446655440000"),
        ("#section", "#section"),
        ("", ""),
        (None, None),
        (123, 123),
    ])
    def test_matrix(self, raw, expected):
        assert sanitize_resource(raw) == expected

    def test_parse_hook(self, tmp_path):
        fp = tmp_path / "db.md"
        fp.write_text(
            "---\ntype: note\ntitle: DB\n"
            "resource: mysql://admin:s3cret@db.internal:3306/app\n---\nBody.\n",
            encoding="utf-8",
        )
        concept, _, _ = parse_source_file(fp, tmp_path)
        assert concept.resource == "mysql://***@db.internal:3306/app"

    def test_parse_hook_absent(self, tmp_path):
        fp = tmp_path / "plain.md"
        fp.write_text("---\ntype: note\ntitle: P\n---\nBody.\n", encoding="utf-8")
        concept, _, _ = parse_source_file(fp, tmp_path)
        assert concept.resource is None


class TestSanitizeLive:
    router: ClassVar[OKFRouter]
    tmp_dir: ClassVar[str]

    @classmethod
    def setup_class(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        cls.router = OKFRouter(
            db_path=os.path.join(cls.tmp_dir, "san.db"),
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

    def test_stored_and_reexported_sanitized(self):
        src = Path(self.tmp_dir) / "cred.md"
        src.write_text(
            "---\ntype: note\ntitle: Cred\n"
            "resource: postgres://u:pw@db.internal:5432/app\n---\nBody.\n",
            encoding="utf-8",
        )
        cid = self.router.import_from_okf(src)
        stored = self.router.get_by_id(cid)
        assert stored.resource == "postgres://***@db.internal:5432/app"

        out = Path(self.tmp_dir) / "exp" / "cred.md"
        self.router.export_mgr.export_to_okf(cid, out)
        assert "postgres://***@db.internal:5432/app" in out.read_text(encoding="utf-8")
        assert "pw@" not in out.read_text(encoding="utf-8")
