"""Bundle-hardening §3: SQLite producer + SourceProducer seam.

Builds a scratch shop database with stdlib sqlite3 (no binary fixture),
produces a bundle, and pins the whole chain: file layout + frontmatter
extras + lint-clean + import edges + error paths + determinism.
"""

import shutil
import sqlite3
import tempfile
from pathlib import Path

import pytest

from okfgraph.components.lint import lint_bundle
from okfgraph.components.producers import (
    PRODUCERS,
    SQLiteProducer,
    producer_for,
)
from okfgraph.router import OKFRouter


def _make_shop_db(path: Path) -> None:
    con = sqlite3.connect(str(path))
    con.executescript(
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT DEFAULT 'nobody@example.com'
        );
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            price REAL
        );
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id),
            product_id INTEGER REFERENCES products(id)
        );
        CREATE TABLE "odd table!" (
            id INTEGER PRIMARY KEY,
            note TEXT
        );
        INSERT INTO users VALUES (1, 'ann', 'a@x'), (2, 'bob', 'b@x');
        INSERT INTO products VALUES (1, 'widget', 9.5);
        INSERT INTO orders VALUES (1, 1, 1), (2, 2, 1);
        """
    )
    con.close()


@pytest.fixture(scope="module")
def shop():
    d = Path(tempfile.mkdtemp())
    db = d / "shop.db"
    _make_shop_db(db)
    out = d / "bundle"
    bundle = SQLiteProducer().produce(db, out)
    yield {"dir": d, "db": db, "out": out, "bundle": bundle}
    shutil.rmtree(d, ignore_errors=True)


def _edges(router):
    rows = router.conn.execute(
        "MATCH (a:Concept)-[:LINKS_TO]->(b:Concept) "
        "RETURN a.id AS src, b.id AS dst"
    ).rows_as_dict().get_all()
    return {(r["src"], r["dst"]) for r in rows}


class TestRegistry:
    def test_sqlite_registered(self):
        assert PRODUCERS["sqlite"] is SQLiteProducer
        assert isinstance(producer_for("sqlite"), SQLiteProducer)

    def test_unknown_producer_names_alternatives(self):
        with pytest.raises(ValueError, match="available.*sqlite"):
            producer_for("docx")


class TestLayout:
    def test_files_written(self, shop):
        assert shop["bundle"].files == 5  # 4 tables + overview
        assert shop["bundle"].producer == "sqlite"
        assert shop["bundle"].source == "shop.db"
        assert (shop["out"] / "database" / "overview.md").is_file()
        for stem in ("users", "products", "orders", "odd_table"):
            assert (shop["out"] / "database" / "tables" / f"{stem}.md").is_file(), stem

    def test_no_index_md_emitted(self, shop):
        # index.md would be skipped on bulk import (reserved name) —
        # the producer must never rely on it.
        assert list((shop["out"] / "database").rglob("index.md")) == []

    def test_frontmatter_extras(self, shop):
        import frontmatter

        post = frontmatter.load(shop["out"] / "database" / "tables" / "orders.md")
        assert post.metadata["type"] == "table"
        assert post.metadata["table"] == "orders"
        assert post.metadata["database"] == "shop.db"
        assert post.metadata["column_count"] == 3
        assert post.metadata["primary_key"] == "id"
        assert post.metadata["row_count"] == 2
        assert "## Schema" in post.content
        assert "## Relationships" in post.content

    def test_fk_links_are_root_relative(self, shop):
        body = (shop["out"] / "database" / "tables" / "orders.md").read_text(
            encoding="utf-8"
        )
        assert "[users](database/tables/users.md)" in body
        assert "[products](database/tables/products.md)" in body

    def test_overview_links_every_table(self, shop):
        body = (shop["out"] / "database" / "overview.md").read_text(encoding="utf-8")
        for stem in ("users", "products", "orders", "odd_table"):
            assert f"(database/tables/{stem}.md)" in body


class TestLintClean:
    def test_produced_bundle_is_lint_clean(self, shop):
        report = lint_bundle(shop["out"])
        assert report["errors"] == []
        assert report["warnings"] == []


class TestImportEdges:
    @pytest.fixture(scope="class")
    @classmethod
    def router(cls, shop):
        r = OKFRouter(
            db_path=str(shop["dir"] / "test_produce.db"),
            bundle_root=str(shop["out"]),
            embedding_dim=512,
            chunk_size=50,
            chunk_overlap=10,
            device="cpu",
        )
        ids = r.import_mgr.import_bundle(shop["out"])
        assert len(ids) == 5, f"expected 5 concepts, got {ids}"
        cls._router = r
        yield cls._router
        cls._router.close()

    def test_fk_edges_exist(self, router):
        edges = _edges(router)
        assert ("database/tables/orders", "database/tables/users") in edges
        assert ("database/tables/orders", "database/tables/products") in edges

    def test_overview_edges_exist(self, router):
        edges = _edges(router)
        for stem in ("users", "products", "orders", "odd_table"):
            assert ("database/overview", f"database/tables/{stem}") in edges

    def test_no_broken_links(self, router):
        assert router.list_broken_links() == []

    def test_no_index_concepts(self, router):
        rows = router.conn.execute(
            "MATCH (c:Concept) WHERE c.id ENDS WITH '/index' "
            "RETURN c.id AS id"
        ).rows_as_dict().get_all()
        assert rows == []


class TestErrors:
    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            SQLiteProducer().produce(tmp_path / "nope.db", tmp_path / "out")

    def test_not_a_database(self, tmp_path):
        fake = tmp_path / "fake.db"
        fake.write_text("this is not sqlite", encoding="utf-8")
        with pytest.raises(ValueError, match="not a SQLite"):
            SQLiteProducer().produce(fake, tmp_path / "out")

    def test_no_user_tables(self, tmp_path):
        empty = tmp_path / "empty.db"
        sqlite3.connect(str(empty)).close()
        with pytest.raises(ValueError, match="no user tables"):
            SQLiteProducer().produce(empty, tmp_path / "out")

    def test_refuses_clobber_without_overwrite(self, shop):
        with pytest.raises(FileExistsError, match="overwrite"):
            SQLiteProducer().produce(shop["db"], shop["out"])

    def test_overwrite_rewrites(self, shop):
        again = SQLiteProducer().produce(shop["db"], shop["out"], overwrite=True)
        assert again.files == shop["bundle"].files

    def test_bad_prefix_rejected(self, shop, tmp_path):
        with pytest.raises(ValueError, match="single directory name"):
            SQLiteProducer().produce(shop["db"], tmp_path, prefix="../evil")


class TestDeterminism:
    def test_byte_identical_rerun(self, shop, tmp_path):
        first = {
            p.relative_to(shop["out"]): p.read_bytes()
            for p in sorted(shop["out"].rglob("*.md"))
        }
        SQLiteProducer().produce(shop["db"], tmp_path / "rerun")
        second = {
            p.relative_to(tmp_path / "rerun"): p.read_bytes()
            for p in sorted((tmp_path / "rerun").rglob("*.md"))
        }
        assert first == second
