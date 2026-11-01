"""WS1: link extraction and source-relative resolution (R29-A, B, C, D).

One case per measured line in R29, so the report is the regression suite.
"""
from __future__ import annotations

from pathlib import Path

from okfgraph.components.diff import state_of_dir
from okfgraph.components.lint import lint_bundle
from okfgraph.components.links import (
    extract_md_links,
    extract_wikilinks,
    link_target,
    normalize_path_link,
)

KNOWN = {
    "architecture/s13-decision-register",
    "architecture/s4-schema",
    "architecture/x",
    "architecture/public-api",
    "Macrame Road to 1.0",
    "flat",
    "releases/x",
}


def _resolve(raw, source, known=KNOWN):
    return link_target(raw, source, known)[0]


# -- R29-A: relative to the source file ------------------------------------

def test_sibling_link_resolves_inside_the_subfolder():
    assert _resolve("s13-decision-register.md", "architecture/x") == \
        "architecture/s13-decision-register"


def test_self_link_resolves_to_itself():
    assert _resolve("x.md", "architecture/x") == "architecture/x"


def test_parent_link_with_percent_encoded_space_is_decoded():
    assert _resolve("../Macrame%20Road%20to%201.0.md", "architecture/x") == \
        "Macrame Road to 1.0"


def test_link_out_of_the_bundle_is_unresolved_but_recorded():
    target, recorded = link_target("../architecture/s4-schema.md", "a", KNOWN)
    assert target is None
    assert recorded == "../architecture/s4-schema"  # never guessed


def test_same_link_resolves_from_a_subfolder_source():
    assert _resolve("../architecture/s4-schema.md", "releases/x") == \
        "architecture/s4-schema"


def test_fallback_to_the_pre_fix_reading_keeps_old_bundles_working():
    # A flat bundle with root-relative links written from a subfolder.
    assert _resolve("flat.md", "sub/x") == "flat"


def test_leading_slash_means_bundle_root():
    assert _resolve("/architecture/x.md", "releases/x") == "architecture/x"


def test_fragment_and_query_are_dropped():
    assert _resolve("x.md#section", "architecture/x") == "architecture/x"
    assert _resolve("x.md?v=2", "architecture/x") == "architecture/x"


# -- R29-B: percent-encoding and angle brackets ----------------------------

def test_angle_bracket_target_with_spaces_is_extracted():
    assert extract_md_links("see [f](<My File.md>)") == ["My File.md"]


def test_link_title_is_accepted():
    assert extract_md_links('see [a](a.md "title")') == ["a.md"]


# -- R29-C: code is never a link ---------------------------------------------

def test_wikilink_inside_inline_code_is_not_a_link():
    assert extract_wikilinks("use `[[bench]]` in the table") == []


def test_fenced_block_is_not_read():
    body = "```\n[[example]] and [x](y.md)\n```\nafter [[real]]"
    assert extract_wikilinks(body) == ["real"]
    assert extract_md_links(body) == []


def test_tilde_fence_is_not_read():
    body = "~~~~\n[x](y.md)\n~~~~\n[z](z.md)"
    assert extract_md_links(body) == ["z.md"]


# -- R29-D: the link pattern is anchored -------------------------------------

def test_two_links_on_one_line_are_two_links():
    body = "[a](public-api.md) and [b](s13-decision-register.md)."
    assert extract_md_links(body) == ["public-api.md", "s13-decision-register.md"]


def test_non_source_targets_are_skipped_and_v1_2_mdx_is_not_mangled():
    assert extract_md_links("[v](v1.2.mdx) [p](pic.png)") == []


def test_txt_and_markdown_extensions_are_sources():
    assert extract_md_links("[api](public-api.txt) [r](README.markdown)") == \
        ["public-api.txt", "README.markdown"]


def test_pre_fix_helper_still_works_without_a_source():
    assert normalize_path_link("./architecture/x.md") == "architecture/x"


# -- End to end: a Macrame-shaped bundle has no broken links -----------------

def _write(root: Path, rel: str, body: str, title: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\ntitle: {title}\n---\n{body}\n", encoding="utf-8")


def test_macrame_shaped_bundle_has_no_broken_links(tmp_path):
    root = tmp_path / "docs"
    _write(root, "Macrame Road to 1.0.md",
           "Start at [s13](architecture/s13-decision-register.md) and the "
           "`[[bench]]` note.", "Macrame Road to 1.0")
    _write(root, "architecture/s13-decision-register.md",
           "Schema: [s4](s4-schema.md). Config: `toml` in a code span.",
           "s13 decision register")
    _write(root, "architecture/s4-schema.md",
           "Back to [road](../Macrame%20Road%20to%201.0.md), see "
           "[api](public-api.txt).", "s4 schema")
    (root / "architecture" / "public-api.txt").write_text(
        "the public api", encoding="utf-8")

    state = state_of_dir(root)
    assert state.broken == set()
    assert ("architecture/s4-schema", "architecture/public-api") in state.edges

    report = lint_bundle(root)
    dangling = [e for e in report["errors"] if e["rule"] == "dangling_link"]
    assert dangling == []
