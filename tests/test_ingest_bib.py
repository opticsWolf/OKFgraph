"""BibTeX ingest end-to-end (model needed for embeddings, like other kinds).

One concept per ``@entry`` under stable ``refs/<key>`` IDs — re-ingesting
the same file upserts instead of duplicating.
"""
from pathlib import Path

import pytest

from okfgraph.errors import OKFError
from okfgraph.router import OKFRouter

BIB_TWO = """@article{smith2020,
  author = {Smith, Jane and Doe, John},
  title = {On graphs},
  journal = {J. Graphs},
  year = {2020},
  abstract = {We study graphs.},
  doi = {10.1/abc},
}
@book{doe2021,
  author = {Doe, John},
  title = {The {DNA} book},
  publisher = {Acme},
  year = {2021},
}
"""


@pytest.fixture()
def router(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    r = OKFRouter(
        db_path=str(tmp_path / "t.db"),
        bundle_root=str(repo),
        device="cpu",
    )
    yield r
    r.close()


def _bib(tmp_path: Path, text: str = BIB_TWO, name: str = "refs.bib") -> str:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


class TestIngestBib:
    def test_two_entries_mint_refs_ids(self, router, tmp_path):
        res = router.ingest_mgr.ingest("bib", bib_path=_bib(tmp_path))
        assert res["concept_ids"] == ["refs/smith2020", "refs/doe2021"]
        assert res["entry_count"] == 2
        assert res["skipped_entries"] == []
        got = router.search_engine.get_by_id("refs/smith2020")
        assert got is not None
        assert got.type == "reference"
        assert got.title == "On graphs"
        assert "Smith" in got.description and "2020" in got.description
        assert set(["reference", "article"]).issubset(set(got.tags))

    def test_reingest_is_stable_upsert(self, router, tmp_path):
        bib = _bib(tmp_path)
        first = router.ingest_mgr.ingest("bib", bib_path=bib)
        second = router.ingest_mgr.ingest("bib", bib_path=bib)
        assert first["concept_ids"] == second["concept_ids"]
        assert router.search_engine.get_by_id("refs/doe2021").title == "The DNA book"

    def test_broken_entries_skipped_valid_imported(self, router, tmp_path):
        bib = _bib(tmp_path, BIB_TWO + "@article{broken, title = {Oops, year = {2022}}\n")
        res = router.ingest_mgr.ingest("bib", bib_path=bib)
        assert res["concept_ids"] == ["refs/smith2020", "refs/doe2021"]
        assert len(res["skipped_entries"]) == 1

    def test_duplicate_keys_disambiguated(self, router, tmp_path):
        text = ("@misc{dup, title = {First}, year = {2020}}\n"
                "@misc{dup, title = {Second}, year = {2021}}\n")
        res = router.ingest_mgr.ingest("bib", bib_path=_bib(tmp_path, text))
        assert res["concept_ids"] == ["refs/dup", "refs/dup~2"]
        assert len(res["skipped_entries"]) == 1
        assert router.search_engine.get_by_id("refs/dup~2").title == "Second"

    def test_file_tags_applied(self, router, tmp_path):
        res = router.ingest_mgr.ingest(
            "bib", bib_path=_bib(tmp_path), tags=["my-lib"])
        got = router.search_engine.get_by_id("refs/smith2020")
        assert "my-lib" in got.tags

    def test_errors_are_typed(self, router, tmp_path):
        with pytest.raises(OKFError) as e:
            router.ingest_mgr.ingest("bib", bib_path=str(tmp_path / "nope.bib"))
        assert e.value.code == "FILE_NOT_FOUND"
        with pytest.raises(OKFError) as e:
            router.ingest_mgr.ingest("bib")
        assert e.value.code == "MISSING_PARAM"
        with pytest.raises(OKFError) as e:
            router.ingest_mgr.ingest(
                "bib", bib_path=_bib(tmp_path), md_path="x.md")
        assert e.value.code == "BAD_VALUE"
        with pytest.raises(OKFError) as e:
            router.ingest_mgr.ingest(
                "bib", bib_path=_bib(tmp_path, "nothing parseable here\n"))
        assert e.value.code == "BAD_VALUE"


def test_cli_bib_args():
    from okfgraph.cli import build_parser

    args = build_parser().parse_args(
        ["ingest", "--kind", "bib", "--bib-path", "refs.bib"])
    assert args.kind == "bib" and args.bib_path == "refs.bib"
