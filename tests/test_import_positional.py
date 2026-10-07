"""Positional import hierarchy (0.10.1 Issue A).

Report scenario: ``okf import /tmp/okf-merge/thoughts/<topic>/*.md`` minted
bare-stem IDs (``20261007084024_0004c1``) instead of
``thoughts/<topic>/<stem>`` — silent misplacement with a success message.

Contract under test (Amendment B §9.1):

- An explicit identity root (``import_file(..., identity_root=DIR)`` / CLI
  ``import --bundle-path DIR FILES``) mints IDs relative to that tree;
  every file must live under it, validated before any import.
- Without one, configured-root lookup is kept and the bare-stem fallback
  stays — but it warns (``hierarchy-dropped``) instead of staying silent.
- Python return shapes are unchanged (``{"concept_id", "images"}``).
"""

import logging
from pathlib import Path

import pytest

from okfgraph.errors import OKFError
from okfgraph.router import OKFRouter


def _write_okf(path: Path, title: str, body: str = "Body text here.") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\ntype: note\ntitle: {title}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def env(tmp_path):
    """Router rooted at ``repo/``; source tree lives outside it (``merge/``).

    Mirrors the incident: positional files outside every configured root.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    merge = tmp_path / "merge"
    topic = merge / "thoughts" / "topic"
    a = _write_okf(topic / "probe_a.md", "Probe A")
    b = _write_okf(topic / "probe_b.md", "Probe B")
    router = OKFRouter(
        db_path=str(tmp_path / "t.db"),
        bundle_root=str(repo),
        device="cpu",
    )
    yield {"router": router, "repo": repo, "merge": merge, "a": a, "b": b}
    router.close()


class TestPositionalIdentityRoot:
    def test_rooted_list_mints_hierarchy(self, env):
        r, merge = env["router"], env["merge"]
        ra = r.import_file(env["a"], identity_root=merge)
        rb = r.import_file(env["b"], identity_root=merge)
        assert ra["concept_id"] == "thoughts/topic/probe_a"
        assert rb["concept_id"] == "thoughts/topic/probe_b"
        assert set(ra) == {"concept_id", "images"}  # return shape unchanged
        assert r.search_engine.get_by_id("thoughts/topic/probe_a") is not None
        assert r.search_engine.get_by_id("thoughts/topic/probe_b") is not None

    def test_outsider_without_root_warns_and_keeps_stem(self, env, caplog):
        r = env["router"]
        with caplog.at_level(logging.WARNING, logger="okfgraph.components.import_"):
            res = r.import_file(env["a"])
        assert res["concept_id"] == "probe_a"  # legacy bare-stem fallback
        assert "hierarchy-dropped" in caplog.text
        assert "probe_a" in caplog.text  # minted ID is named

    def test_outsider_of_explicit_root_rejected_before_import(self, env):
        r = env["router"]
        elsewhere = env["repo"] / "other"
        elsewhere.mkdir()
        with pytest.raises(OKFError) as exc:
            r.import_file(env["a"], identity_root=elsewhere)
        assert exc.value.code == "BAD_VALUE"
        # Nothing imported — the refusal precedes every write.
        assert r.search_engine.get_by_id("probe_a") is None
        assert r.search_engine.get_by_id("thoughts/topic/probe_a") is None

    def test_lone_outsider_still_imports(self, env):
        r = env["router"]
        res = r.import_file(env["b"])
        assert res["concept_id"] == "probe_b"
        assert r.search_engine.get_by_id("probe_b") is not None

    def test_bulk_path_unchanged(self, env):
        r, merge = env["router"], env["merge"]
        ids = r.import_mgr.import_bundle(bundle_path=merge)
        assert sorted(ids) == ["thoughts/topic/probe_a", "thoughts/topic/probe_b"]
