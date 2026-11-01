"""``ingest("thoughts", folder=...)`` writes the thought into a bundle
folder and imports it: ``concept_id`` + ``materialize`` without naming
the file."""

import hashlib

import pytest

from okfgraph.cli import build_parser
from okfgraph.errors import OKFError
from okfgraph.router import OKFRouter


def _vec(text, dim):
    h = hashlib.sha256((text or "").encode()).digest()
    v = [((h[i % len(h)] / 255.0) - 0.5) for i in range(dim)]
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


@pytest.fixture
def env(tmp_path, monkeypatch):
    from okfgraph.components import embedding as E

    dim = lambda self: int(getattr(self, "_truncate_dim", 512) or 512)  # noqa: E731
    monkeypatch.setattr(E.LazyRustEncoder, "encode",
                        lambda self, text, task="Document": _vec(text, dim(self)))
    monkeypatch.setattr(E.LazyRustEncoder, "encode_batch",
                        lambda self, texts, task="Document": [_vec(t, dim(self)) for t in texts])
    monkeypatch.setattr(E.LazyRustEncoder, "count_tokens",
                        lambda self, text: len((text or "").split()))
    monkeypatch.setattr(E.LazyRustEncoder, "is_loaded", property(lambda self: True))
    kb, notes = tmp_path / "kb", tmp_path / "notes"
    kb.mkdir()
    notes.mkdir()
    r = OKFRouter(db_path=str(tmp_path / "kb.db"), bundle_root=str(kb),
                  roots={"notes": str(notes)}, embedding_dim=512, device="cpu")
    yield r, kb, notes
    r.close()


def _think(r, **kw):
    return r.ingest("thoughts", thoughts="Per-file hashing beats directory hashing.",
                    topic="perf", **kw)


def test_default_thought_is_graph_only(env):
    r, kb, notes = env
    out = _think(r, title="No file")
    assert r.get_by_id(out["concept_id"]) is not None
    assert not out.get("materialized")
    assert not list(kb.rglob("*.md")) and not list(notes.rglob("*.md"))


def test_titled_thought_lands_in_the_folder(env):
    r, kb, _ = env
    out = _think(r, folder="decisions/perf", title="Cache per file")
    assert out["concept_id"] == "decisions/perf/cache-per-file"
    assert out["materialized"] is True
    path = kb / "decisions" / "perf" / "cache-per-file.md"
    assert path.is_file() and "Per-file hashing" in path.read_text(encoding="utf-8")
    assert r.get_by_id("decisions/perf/cache-per-file").title == "Cache per file"


def test_same_title_same_folder_is_one_concept(env):
    r, kb, _ = env
    _think(r, folder="decisions", title="Cache per file")
    out = r.ingest("thoughts", thoughts="Revised: hash per file, batch per dir.",
                   topic="perf", folder="decisions", title="Cache per file")
    assert out["concept_id"] == "decisions/cache-per-file"
    assert list((kb / "decisions").glob("*.md")) == [kb / "decisions" / "cache-per-file.md"]
    assert "Revised" in (kb / "decisions" / "cache-per-file.md").read_text(encoding="utf-8")


def test_untitled_thoughts_never_collide(env):
    r, kb, _ = env
    a = _think(r, folder="inbox")["concept_id"]
    b = _think(r, folder="inbox/")["concept_id"]
    assert a != b and a.startswith("inbox/") and b.startswith("inbox/")
    assert len(list((kb / "inbox").glob("*.md"))) == 2


def test_alias_folder_writes_into_that_root(env):
    r, kb, notes = env
    out = _think(r, folder="@notes/perf", title="Hash budget", related=["missing/x"])
    assert out["concept_id"] == "@notes/perf/hash-budget"
    assert (notes / "perf" / "hash-budget.md").is_file()
    assert not (kb / "perf").exists()
    assert out["unresolved_related"] == ["missing/x"]


@pytest.mark.parametrize("kw", [
    {"folder": "../escape"},
    {"folder": "/abs"},
    {"folder": "a/../b"},
    {"folder": ""},
    {"folder": "has space"},
    {"folder": "@nope/x"},
    {"folder": "d", "concept_id": "d/x"},
    {"folder": "d", "materialize": False},
])
def test_bad_folder_refused(env, kw):
    r, kb, _ = env
    with pytest.raises(OKFError) as exc:
        _think(r, title="T", **kw)
    assert exc.value.code == "BAD_VALUE"
    assert not any(p.suffix == ".md" for p in kb.parent.rglob("*.md"))


def test_folder_is_thoughts_only(env):
    r, kb, _ = env
    (kb / "a.md").write_text("---\ntitle: A\n---\n\nbody\n", encoding="utf-8")
    with pytest.raises(OKFError) as exc:
        r.ingest("md", md_path=str(kb / "a.md"), folder="x")
    assert exc.value.code == "BAD_VALUE" and exc.value.fields["ignored"] == ["folder"]


def test_cli_flag():
    args = build_parser().parse_args(
        ["ingest", "--kind", "thoughts", "--topic", "perf", "--thoughts", "t",
         "--folder", "decisions/perf"])
    assert args.folder == "decisions/perf"
