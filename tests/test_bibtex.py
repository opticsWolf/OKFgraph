"""BibTeX parser unit tests (no model, no DB — pure stdlib parsing)."""
from okfgraph.components.bibtex import (
    BibFile,
    parse_bibtex,
    slugify_key,
    unwrap_braces,
)


def test_basic_entry():
    bib = parse_bibtex("""@article{smith2020,
      author = {Smith, Jane and Doe, John},
      title = {On graphs},
      year = {2020},
    }""")
    assert isinstance(bib, BibFile)
    assert bib.skipped == []
    assert len(bib.entries) == 1
    e = bib.entries[0]
    assert (e.entrytype, e.key) == ("article", "smith2020")
    assert e.fields["author"] == "Smith, Jane and Doe, John"
    assert e.fields["year"] == "2020"


def test_nested_braces_and_quotes():
    bib = parse_bibtex("""@inproceedings{doe21,
      title = {The {DNA} of {Graph} Neural Networks},
      booktitle = "Proc. of the Conf. on Graphs",
      year = 2021,
    }""")
    (e,) = bib.entries
    assert e.fields["title"] == "The {DNA} of {Graph} Neural Networks"
    assert e.fields["booktitle"] == "Proc. of the Conf. on Graphs"
    assert unwrap_braces(e.fields["title"]) == "The DNA of Graph Neural Networks"


def test_concat_and_string_macros():
    bib = parse_bibtex("""@string{jmlr = {Journal of Machine Learning Research}}
    @article{key1,
      journal = jmlr # {, Special Issue},
      year = {2022},
    }""")
    (e,) = bib.entries
    assert e.fields["journal"] == "Journal of Machine Learning Research, Special Issue"


def test_comments_preamble_skipped():
    bib = parse_bibtex("""% a leading comment
    @comment{anything goes here, even { unbalanced}
    @preamble{"\\newcommand{\\x}{y}"}
    @book{ok2019, title = {Fine}, year = {2019}} % trailing comment
    """)
    assert [e.key for e in bib.entries] == ["ok2019"]
    assert bib.skipped == []


def test_paren_delimiters():
    bib = parse_bibtex("@misc(key2, title = {Parens}, year = {2023})")
    (e,) = bib.entries
    assert (e.entrytype, e.key) == ("misc", "key2")


def test_malformed_entries_recovered_not_fatal():
    bib = parse_bibtex("""@article{good1, title = {Fine}, year = {2020}}
    @article{broken1, title = {Never closed, year = {2021}}
    @article{, title = {No key}}
    @article{good2, title = {Also fine}, year = {2022}}
    """)
    assert [e.key for e in bib.entries] == ["good1", "good2"]
    assert len(bib.skipped) == 2
    assert all("line" in s and "reason" in s for s in bib.skipped)


def test_percent_inside_value_is_data():
    bib = parse_bibtex('@misc{p1, note = {Grew by 50% in a year}, year = {2024}}')
    (e,) = bib.entries
    assert e.fields["note"] == "Grew by 50% in a year"


def test_slugify_key():
    assert slugify_key("smith2020") == "smith2020"
    assert slugify_key("Doe:2021a") == "Doe_2021a"
    assert slugify_key("a/b\\c") == "a_b_c"
    assert slugify_key("../../evil") == "evil"
    assert slugify_key("") == "untitled"
