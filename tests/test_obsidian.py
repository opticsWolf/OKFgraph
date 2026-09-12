"""Obsidian-vault compatibility: name-based [[wikilink]] import, ambiguous
names that never resolve, wiki-aware repair, and the obsidian export flavor
with a lossless edge round-trip."""

import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from okfgraph.components.links import build_name_index, resolve_wiki
from okfgraph.router import OKFRouter

FIX = Path(__file__).parent / "fixtures"


def _edges(router):
    rows = router.conn.execute(
        "MATCH (a:Concept)-[:LINKS_TO]->(b:Concept) "
        "RETURN a.id AS src, b.id AS dst"
    ).rows_as_dict().get_all()
    return {(r["src"], r["dst"]) for r in rows}


class TestWikiLinksPure:
    def test_precedence_uid_alias_title_stem(self):
        concepts = [
            {"id": "a/x", "title": "Shared", "uid": "shared"},
            {"id": "b/y", "title": "Other", "aliases": ["shared"]},
        ]
        maps, ambiguous = build_name_index(concepts)
        known = {"a/x", "b/y"}
        # uid wins over alias over title even on the same key
        assert resolve_wiki("shared", maps, known) == "a/x"
        assert resolve_wiki("Shared", maps, known) == "a/x"
        assert resolve_wiki("other", maps, known) == "b/y"
        assert resolve_wiki("y", maps, known) == "b/y"  # stem
        assert ambiguous == set()

    def test_ambiguity_resolves_to_nothing(self):
        concepts = [
            {"id": "a", "title": "Duplicated Name"},
            {"id": "b", "title": "duplicated name"},
        ]
        maps, ambiguous = build_name_index(concepts)
        assert ambiguous == {"title:duplicated name"}
        assert resolve_wiki("Duplicated Name", maps, {"a", "b"}) is None

    def test_exact_id_wins(self):
        concepts = [{"id": "docs/guide", "title": "Guide"}]
        maps, _ = build_name_index(concepts)
        assert resolve_wiki("docs/guide", maps, {"docs/guide"}) == "docs/guide"
        assert resolve_wiki("guide", maps, {"docs/guide"}) == "docs/guide"

    def test_fragments_and_display_stripped(self):
        concepts = [{"id": "a", "title": "Target"}]
        maps, _ = build_name_index(concepts)
        assert resolve_wiki("Target#section", maps, {"a"}) == "a"


class TestObsidianVaultLive:
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
            db_path=str(Path(tmp_dir) / "test_obsidian.db"),
            bundle_root=str(FIX / "obsidian_vault"),
            embedding_dim=512,
            chunk_size=50,
            chunk_overlap=10,
            device="cpu",
        )
        ids = r.import_mgr.import_bundle(FIX / "obsidian_vault")
        assert len(ids) == 5, f"expected 5 concepts, got {ids}"
        cls._router = r
        yield cls._router
        cls._router.close()

    def test_name_links_resolve(self, router):
        edges = _edges(router)
        # [[Honey Badger]] by title, [[fauna|..]] by alias (self-link)
        assert ("wildlife", "sub/honey-badger") in edges
        assert ("wildlife", "wildlife") in edges
        # Backlink by title survives the subdirectory (rename-safe)
        assert ("sub/honey-badger", "wildlife") in edges

    def test_ambiguous_name_stays_broken(self, router):
        broken = {(b["source"], b["target"]) for b in router.list_broken_links()}
        assert ("ambig-ref", "Duplicated Name") in broken
        edges = _edges(router)
        assert not any(s == "ambig-ref" and d.startswith("dupe") for s, d in edges)

    def test_repair_never_guesses_ambiguous(self, router):
        assert router.repair_links() == 0
        broken = {(b["source"], b["target"]) for b in router.list_broken_links()}
        assert ("ambig-ref", "Duplicated Name") in broken

    def test_wiki_repair_when_target_appears(self, router, tmp_path):
        (tmp_path / "late.md").write_text(
            "---\ntitle: Early Bird\ntype: note\n---\n\nSee [[Late Arrival]].\n"
        )
        cid = router.import_from_okf(tmp_path / "late.md")
        assert any(t == "Late Arrival" for _, t in
                   ((b["source"], b["target"]) for b in router.list_broken_links()))
        (tmp_path / "arrival.md").write_text(
            "---\ntitle: Late Arrival\ntype: note\n---\n\nHello.\n"
        )
        router.import_from_okf(tmp_path / "arrival.md")
        assert router.repair_links() == 1
        assert (cid, "arrival") in _edges(router)

    def test_obsidian_export_round_trip(self, router, tmp_path):
        out = tmp_path / "vault-out"
        ids = router.export_mgr.export_bundle(out, flavor="obsidian")
        assert ids
        texts = [p.read_text(encoding="utf-8") for p in out.rglob("*.md")]
        assert any("[[" in t for t in texts), "expected [[wikilinks]] in export"
        assert not list(out.rglob("index.md")), "vaults carry no index.md"
        before = _edges(router)

        r2 = OKFRouter(
            db_path=str(tmp_path / "roundtrip.db"),
            bundle_root=str(out),
            embedding_dim=512,
            chunk_size=50,
            chunk_overlap=10,
            device="cpu",
        )
        try:
            r2.import_mgr.import_bundle(out)
            rows = r2.conn.execute(
                "MATCH (a:Concept)-[:LINKS_TO]->(b:Concept) "
                "RETURN a.id AS src, b.id AS dst"
            ).rows_as_dict().get_all()
            after = {(r["src"], r["dst"]) for r in rows}
        finally:
            r2.close()
        assert after == before

    def test_frontmatter_id_round_trip(self, router, tmp_path):
        (tmp_path / "keyed.md").write_text(
            "---\ntitle: Keyed Concept\ntype: note\nid: stable-key-123\n---\n\nBody.\n"
        )
        cid = router.import_from_okf(tmp_path / "keyed.md")
        single = tmp_path / "single"
        single.mkdir()
        router.export_mgr.export_to_okf(cid, single / "keyed.md", flavor="obsidian")
        exported = (single / "keyed.md").read_text(encoding="utf-8")
        assert "id: stable-key-123" in exported
        assert "uid:" not in exported
