"""scripts/eval_search.py: metrics, label/config parsing, and one stubbed
end-to-end run (no model weights)."""

import hashlib
import json
import math

import pytest

from scripts import eval_search as ev


def _vec(text, dim):
    h = hashlib.sha256((text or "").encode()).digest()
    v = [((h[i % len(h)] / 255.0) - 0.5) for i in range(dim)]
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


# ── metrics ─────────────────────────────────────────────────────────────

def test_perfect_and_empty_rankings():
    rel = {"a": 2, "b": 1}
    assert ev.score_ranking(["a", "b", "x"], rel, 10)["ndcg"] == pytest.approx(1.0)
    m = ev.score_ranking(["x", "y"], rel, 10)
    assert (m["ndcg"], m["mrr"], m["recall"]) == (0.0, 0.0, 0.0)


def test_repeats_score_once_and_cost_a_slot():
    rel = {"a": 1, "b": 1}
    once = ev.score_ranking(["a", "b"], rel, 2)
    twice = ev.score_ranking(["a", "a"], rel, 2)
    assert once["recall"] == 1.0 and twice["recall"] == 0.5
    assert twice["distinct"] == 0.5 and twice["ndcg"] < once["ndcg"]


def test_graded_ndcg_and_mrr():
    rel = {"a": 2, "b": 1}
    m = ev.score_ranking(["b", "a"], rel, 10)
    dcg = 1 / math.log2(2) + 3 / math.log2(3)
    idcg = 3 / math.log2(2) + 1 / math.log2(3)
    assert m["ndcg"] == pytest.approx(dcg / idcg) and m["mrr"] == 1.0
    assert ev.score_ranking(["x", "b"], rel, 10)["mrr"] == 0.5
    assert ev.score_ranking(["x", "b"], rel, 1)["recall"] == 0.0


# ── parsing ─────────────────────────────────────────────────────────────

def test_parse_config():
    c = ev.parse_config("cross:mmarco@q8+dedupe")
    assert (c.base, c.model, c.precision, c.dedupe) == (
        "cross", ev.MODELS["mmarco"], "q8", True)
    c = ev.parse_config("cross@q8")
    assert (c.model, c.precision) == (ev.MODELS["minilm"], "q8")
    assert ev.parse_config("cross").precision == "fp32"
    assert ev.parse_config("cross:org/model").model == "org/model"
    assert ev.parse_config("hub").search_kwargs("chunks") == {
        "target": "chunks", "hub_rerank": True}
    assert ev.parse_config("cross").search_kwargs("concepts") == {
        "target": "concepts", "rank": "cross"}
    assert ev.parse_config("rrf+dedupe").search_kwargs("concepts") is None
    for bad in ("bm25", "hub:x", "rrf@q8", "cross@fp16"):
        with pytest.raises(ValueError):
            ev.parse_config(bad)


def test_load_labels(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text("\n".join([
        "// comment",
        json.dumps({"id": "a", "query": "q", "relevant": ["x", "y"], "split": "dev"}),
        json.dumps({"query": "q2", "relevant": {"x": 2}, "target": "both",
                    "lang": "de", "split": "heldout"}),
    ]), encoding="utf-8")
    labels = ev.load_labels(str(p))
    assert labels[0].relevant == {"x": 1, "y": 1} and labels[0].targets == ["chunks"]
    assert labels[1].targets == ["chunks", "concepts"] and labels[1].lang == "de"
    assert [lb.id for lb in ev.load_labels(str(p), "heldout")] == ["q003"]


@pytest.mark.parametrize("row", [
    {"query": "q"},
    {"query": "q", "relevant": {}},
    {"query": "q", "relevant": {"x": -1}},
    {"query": "q", "relevant": ["x"], "target": "images"},
])
def test_bad_labels(tmp_path, row):
    p = tmp_path / "l.jsonl"
    p.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError):
        ev.load_labels(str(p))


# ── end to end ──────────────────────────────────────────────────────────

class FakeSession:
    max_length = 512
    used_cuda = False
    precision = "fp32"

    def count_tokens(self, text):
        return len(text.split())

    def score_batch(self, pairs):
        return [float(sum(d.lower().split().count(w) for w in q.lower().split()))
                for q, d in pairs]

    def score_long(self, query, doc, window=None, stride=None):
        return self.score_batch([(query, doc)])[0]


def test_end_to_end(tmp_path, monkeypatch):
    import embroider

    from okfgraph.components import embedding as E

    dim = lambda self: int(getattr(self, "_truncate_dim", 512) or 512)  # noqa: E731
    monkeypatch.setattr(E.LazyRustEncoder, "encode",
                        lambda self, text, task="Document": _vec(text, dim(self)))
    monkeypatch.setattr(E.LazyRustEncoder, "encode_batch",
                        lambda self, texts, task="Document": [_vec(t, dim(self)) for t in texts])
    monkeypatch.setattr(E.LazyRustEncoder, "count_tokens",
                        lambda self, text: len((text or "").split()))
    monkeypatch.setattr(E.LazyRustEncoder, "is_loaded", property(lambda self: True))
    opened = []
    monkeypatch.setattr(embroider.Reranker, "open", staticmethod(
        lambda model, **kw: opened.append((model, kw.get("precision"))) or FakeSession()))

    kb = tmp_path / "kb"
    kb.mkdir()
    for name, body in {"cats": "cats purr and cats nap", "dogs": "dogs bark",
                       "birds": "birds sing, cats watch"}.items():
        (kb / f"{name}.md").write_text(f"---\ntitle: {name}\n---\n\n{body}\n",
                                       encoding="utf-8")
    from okfgraph.router import OKFRouter

    r = OKFRouter(db_path=str(tmp_path / "kb.db"), bundle_root=str(kb),
                  embedding_dim=512, device="cpu")
    r.ort_dylib = r.ort_dylib or "stub"
    r.import_bundle(str(kb))
    r.close()

    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join(json.dumps(x) for x in [
        {"id": "q1", "query": "cats", "relevant": {"cats": 2, "birds": 1},
         "target": "both", "lang": "en"},
        {"id": "q2", "query": "dogs bark", "relevant": ["dogs"], "lang": "de"},
    ]), encoding="utf-8")
    md, js = tmp_path / "r.md", tmp_path / "r.json"
    monkeypatch.setattr(ev, "open_router", lambda db, copy: _open(db))

    def _open(db):
        router = OKFRouter(db_path=db, bundle_root=str(kb), embedding_dim=512,
                           device="cpu")
        router.ort_dylib = router.ort_dylib or "stub"
        return router

    code = ev.main([str(labels), "--db", str(tmp_path / "kb.db"),
                    "--configs", "rrf,hub,cross,cross:mmarco@q8,rrf+dedupe",
                    "--md", str(md), "--json", str(js)])
    assert code == 0
    report = json.loads(js.read_text(encoding="utf-8"))
    by = {c["config"]: c for c in report["summary"]["configs"]}
    assert report["summary"]["baseline"] == "rrf"
    assert by["rrf"]["n"] == 3 and by["rrf+dedupe"]["n"] == 2
    assert by["cross"]["ndcg"] == pytest.approx(1.0)
    assert by["cross"]["long_share"] == 0.0
    assert set(by["cross"]["slices"]) == {"chunks/en", "chunks/de", "concepts/en"}
    assert opened == [(ev.MODELS["minilm"], None), (ev.MODELS["mmarco"], "q8")]
    text = md.read_text(encoding="utf-8")
    assert "| `cross:mmarco@q8` | 3 |" in text and "by slice" in text


def test_failed_config_is_reported_not_fatal(tmp_path, monkeypatch):
    class Boom:
        search_engine = type("E", (), {"reranker": None})()
        reranker = None

        def search(self, *a, **kw):
            raise RuntimeError("no weights")

    run = ev.run_config(Boom(), ev.parse_config("rrf"),
                        [ev.Label("q", "x", {"a": 1}, ["chunks"])], 10, {})
    assert run.error == "RuntimeError: no weights"
    summary = ev.summarize([run])
    assert summary["baseline"] is None and summary["configs"][0]["error"]
    assert "error: RuntimeError" in ev.markdown(summary, {
        "labels": "l", "db": "d", "k": 10, "queries": 1})
