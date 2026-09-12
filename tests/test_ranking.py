"""Pure ranking tests: seeds + exact PPR (no database, no model).

The golden fixture (tests/fixtures/ppr_graph.*.json) locks the cascade
output: any determinism regression fails here first.
"""

import json
from pathlib import Path

import pytest

from okfgraph.components.ranking import (
    STOPWORDS,
    ppr,
    query_tokens,
    seed_ranked_ppr,
    seeds,
)

FIX = Path(__file__).parent / "fixtures"


def _concepts():
    return [
        {"id": "a", "title": "Honey Badger Defense", "description": "thick skin",
         "tags": ["wildlife"], "body": "fearless defense"},
        {"id": "b", "title": "Savanna Food Web", "description": "predators",
         "tags": [], "body": "links honey badger and lion"},
        {"id": "c", "title": "Lion Prides", "description": "hunting",
         "tags": [], "body": "savanna hunters"},
        {"id": "d", "title": "Quantum Tunneling", "description": "physics",
         "tags": [], "body": "barriers"},
    ]


class TestQueryTokens:
    def test_stopwords_and_short_tokens_dropped(self):
        assert query_tokens("the honey badger and ox") == ["badger", "honey"]
        assert "the" not in query_tokens("the")

    def test_stopword_list_stable(self):
        assert len(STOPWORDS) == 37


class TestSeeds:
    def test_title_beats_meta_beats_body(self):
        # a: title x3 + body x1 = 10; b: body x2 = 2
        assert seeds(_concepts(), "honey badger defense") == [("a", 10.0), ("b", 2.0)]

    def test_tags_count_as_meta(self):
        concepts = [{"id": "x", "title": "Other", "description": "",
                     "tags": ["wildlife"], "body": "nothing"}]
        assert seeds(concepts, "wildlife") == [("x", 2.0)]

    def test_ties_break_by_id(self):
        concepts = [
            {"id": "b", "title": "Shared", "description": "", "tags": [], "body": ""},
            {"id": "a", "title": "Shared", "description": "", "tags": [], "body": ""},
        ]
        assert [cid for cid, _ in seeds(concepts, "shared")] == ["a", "b"]

    def test_empty_query_gives_no_seeds(self):
        assert seeds(_concepts(), "the and") == []
        assert seeds(_concepts(), "") == []

    def test_k_truncates(self):
        assert len(seeds(_concepts(), "honey badger savanna lion", k=1)) == 1


class TestPpr:
    def test_golden_fixture(self):
        fix = json.loads((FIX / "ppr_graph.json").read_text())
        expected = json.loads((FIX / "ppr_graph.expected.json").read_text())
        got = seed_ranked_ppr(fix["concepts"], [tuple(e) for e in fix["edges"]], fix["query"])
        assert [[cid, score] for cid, score in got] == expected

    def test_deterministic_across_runs(self):
        c, e = _concepts(), [("a", "b"), ("b", "c"), ("c", "d")]
        assert seed_ranked_ppr(c, e, "honey badger") == seed_ranked_ppr(c, e, "honey badger")

    def test_mass_conserved(self):
        got = seed_ranked_ppr(_concepts(), [("a", "b"), ("b", "c")], "honey badger")
        assert sum(s for _, s in got) == pytest.approx(1.0)

    def test_unknown_start_raises(self):
        with pytest.raises(KeyError):
            ppr(["a"], [], ["missing"])

    def test_bad_weights_raise(self):
        with pytest.raises(ValueError):
            ppr(["a", "b"], [], ["a", "b"], weights=[1.0])

    def test_sparse_graph_degrades_to_seed_order(self):
        # No edges: PPR == normalized seed distribution == seed order.
        got = seed_ranked_ppr(_concepts(), [], "honey badger defense")
        assert [cid for cid, _ in got] == ["a", "b"]

    def test_dangling_mass_returns_to_seeds(self):
        got = dict(ppr(["a", "b"], [], ["a"], k=None))
        assert got["a"] == pytest.approx(1.0)
        assert "b" not in got
