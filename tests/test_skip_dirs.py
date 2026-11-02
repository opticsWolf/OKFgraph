"""The shared dot/tool-dir skip rule: anchored at the walk root (2026-11-02).

The 2026-11-02 finding: a bundle whose root *sits under* a dot-directory
(`.../OKFgraph/.okf-accept-1102/tree`) had every file classified as
"hidden/tool dirs" because `in_skipped_dir` inspected absolute path
parts — the root's own ancestors counted. The rule now anchors at the
walk root: dot/tool dirs *inside* the bundle stay hidden by design;
ancestors of the root never count.
"""
from pathlib import Path

from okfgraph.components.import_ import (
    count_skipped_concept_files,
    in_skipped_dir,
)

CONCEPT = (
    "---\nid: x\n---\n\nbody\n"
)


def test_root_ancestors_never_hide(tmp_path):
    tree = tmp_path / ".okf-scratch" / "tree"
    tree.mkdir(parents=True)
    f = tree / "note.md"
    f.write_text(CONCEPT, encoding="utf-8")
    assert f.is_file()
    # anchored at the root: the dot-ancestor is invisible to the rule
    assert in_skipped_dir(f, root=tree) is False
    assert count_skipped_concept_files(tree) == 0


def test_dot_dirs_inside_root_still_hide(tmp_path):
    tree = tmp_path / "tree"
    (tree / ".git").mkdir(parents=True)
    (tree / "notes").mkdir()
    hidden = tree / ".git" / "leak.md"
    hidden.write_text(CONCEPT, encoding="utf-8")
    kept = tree / "notes" / "note.md"
    kept.write_text(CONCEPT, encoding="utf-8")
    assert in_skipped_dir(hidden, root=tree) is True
    assert in_skipped_dir(kept, root=tree) is False
    assert count_skipped_concept_files(tree) == 1


def test_tool_dirs_inside_root_still_hide(tmp_path):
    tree = tmp_path / "tree"
    (tree / "node_modules" / "pkg").mkdir(parents=True)
    f = tree / "node_modules" / "pkg" / "README.md"
    f.write_text(CONCEPT, encoding="utf-8")
    assert in_skipped_dir(f, root=tree) is True


def test_root_relative_root_ancestors_ok(tmp_path):
    # relative root + absolute fp: relative_to fails (ValueError) and the
    # caller-safe fallback recomputes without the root — still no crash
    tree = tmp_path / "tree"
    tree.mkdir()
    f = tree / "note.md"
    f.write_text(CONCEPT, encoding="utf-8")
    assert in_skipped_dir(f, root=tree.name) is False


def test_legacy_call_unchanged():
    p = Path("D:/repo/.git/x.md")
    assert in_skipped_dir(p) is True
    assert in_skipped_dir(Path("D:/repo/tree/x.md"), root=Path("D:/repo")) is False
