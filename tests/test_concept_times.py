"""Graph-kept concept times and the opt-in recency boost (0.11.0).

``created_at`` survives re-imports, ``updated_at`` moves only when the
content changes, ``effective_at`` is the authored date (``timestamp`` >
``updated`` > ``date`` > ``created``) or else ``updated_at``. ``since`` /
``until`` and ``recency`` read ``effective_at``.
"""

import hashlib
from datetime import datetime, timedelta

import pytest

from okfgraph.cli import _search_filters, build_parser
from okfgraph.components import timestamps as T
from okfgraph.errors import OKFError
from okfgraph.router import OKFRouter

DOCS = {
    "plain": "",
    "dated": "date: 2024-01-02\n",
    "both": "date: 2020-05-05\nupdated: 2023-03-04T10:00:00Z\n",
    "stamped": "timestamp: 2022-02-02T00:00:00\nupdated: 2023-03-04\n",
}


def _vec(text, dim):
    h = hashlib.sha256((text or "").encode()).digest()
    v = [((h[i % len(h)] / 255.0) - 0.5) for i in range(dim)]
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from okfgraph.components import embedding as E

    mp = pytest.MonkeyPatch()
    dim = lambda self: int(getattr(self, "_truncate_dim", 512) or 512)  # noqa: E731
    mp.setattr(E.LazyRustEncoder, "encode",
               lambda self, text, task="Document": _vec(text, dim(self)))
    mp.setattr(E.LazyRustEncoder, "encode_batch",
               lambda self, texts, task="Document": [_vec(t, dim(self)) for t in texts])
    mp.setattr(E.LazyRustEncoder, "count_tokens",
               lambda self, text: len((text or "").split()))
    mp.setattr(E.LazyRustEncoder, "is_loaded", property(lambda self: True))

    base = tmp_path_factory.mktemp("times")
    kb = base / "kb"
    kb.mkdir()
    for cid, fm in DOCS.items():
        (kb / f"{cid}.md").write_text(
            f"---\ntitle: {cid}\ntype: note\n{fm}---\n\nwombat {cid}\n",
            encoding="utf-8")
    r = OKFRouter(db_path=str(base / "kb.db"), bundle_root=str(kb),
                  embedding_dim=512, device="cpu")
    r.import_bundle()
    yield r, kb
    r.close()
    mp.undo()


def _times(r, cid):
    row = r.conn.execute(
        "MATCH (c:Concept {id: $id}) RETURN c.created_at AS c, "
        "c.updated_at AS u, c.effective_at AS e, c.content_sig AS s",
        {"id": cid}).rows_as_dict().get_all()[0]
    return row


def test_every_concept_gets_times(env):
    r, _ = env
    for cid in DOCS:
        t = _times(r, cid)
        assert t["c"] is not None and t["u"] is not None and t["s"]
    assert _times(r, "plain")["e"] == _times(r, "plain")["u"]


def test_authored_dates_win_in_order(env):
    r, _ = env
    assert _times(r, "dated")["e"] == datetime(2024, 1, 2)     # YAML date imports
    assert _times(r, "both")["e"] == datetime(2023, 3, 4, 10)  # updated > date, UTC
    assert _times(r, "stamped")["e"] == datetime(2022, 2, 2)   # timestamp wins


def test_reimport_keeps_created_and_moves_updated_on_change(env):
    r, kb = env
    before = _times(r, "plain")
    r.import_mgr.import_from_okf(kb / "plain.md", force=True)
    same = _times(r, "plain")
    assert (same["c"], same["u"], same["s"]) == (before["c"], before["u"], before["s"])

    (kb / "plain.md").write_text(
        "---\ntitle: plain\ntype: note\n---\n\nwombat plain, edited\n", encoding="utf-8")
    r.import_mgr.import_from_okf(kb / "plain.md", force=True)
    after = _times(r, "plain")
    assert after["c"] == before["c"]
    assert after["u"] > before["u"] and after["s"] != before["s"]
    assert after["e"] == after["u"]


def test_read_shows_times_export_omits_them(env, tmp_path):
    r, _ = env
    concept = r.get_by_id("dated")
    data = concept.public_dict()
    assert data["created_at"] and data["effective_at"] == datetime(2024, 1, 2)
    assert "content_sig" not in data
    meta, _ = concept.export_frontmatter()
    assert not set(meta) & {"created_at", "updated_at", "effective_at", "content_sig"}
    assert meta["date"] == "2024-01-02"


def test_since_until_use_effective_time(env):
    r, _ = env
    rows = r.search("wombat", since="2023-01-01", until="2024-06-01", exclude_tags=[])
    assert {x["id"] for x in rows} == {"dated", "both"}
    recent = r.search("wombat", since=datetime.now() - timedelta(days=1), exclude_tags=[])
    assert {x["id"] for x in recent} == {"plain"}


def test_recency_reorders_by_effective_time(env, monkeypatch):
    r, _ = env
    eng = r.search_engine
    order = ["stamped", "both", "dated", "plain"]  # oldest first by relevance

    def fake(table, vec_index, fts_index, query_vec, query, k):
        return [(cid, 1.0 / (60 + i)) for i, cid in enumerate(order)], False

    monkeypatch.setattr(eng, "_rrf_candidates", fake)
    plain = r.search("wombat", exclude_tags=[])
    assert [x["id"] for x in plain] == order
    boosted = r.search("wombat", recency=30, recency_weight=0.9, exclude_tags=[])
    assert boosted[0]["id"] == "plain"
    assert boosted[0]["recency_score"] > 0.9
    assert all(0 <= x["recency_score"] <= 1 for x in boosted)
    page2 = r.search("wombat", recency=30, recency_weight=0.9, limit=2, offset=2,
                     exclude_tags=[])
    assert [x["id"] for x in page2] == [x["id"] for x in boosted[2:4]]


def test_recency_on_chunks_and_ppr(env):
    r, _ = env
    rows = r.search("wombat", target="chunks", recency=7, exclude_tags=[])
    assert rows and all("recency_score" in x for x in rows)
    rows = r.search("wombat", rank="ppr", recency=7, exclude_tags=[])
    assert all("recency_score" in x for x in rows)


@pytest.mark.parametrize("kwargs", [
    {"recency": 0}, {"recency": -3}, {"recency": True}, {"recency": "30"},
    {"recency": 30, "recency_weight": 1.5},
    {"recency_weight": 0.5},
    {"recency": 30, "target": "images"},
    {"recency": 30, "target": "chunks", "expand": True},
])
def test_recency_validation(env, kwargs):
    r, _ = env
    with pytest.raises(OKFError) as exc:
        r.search("wombat", **kwargs)
    assert exc.value.code == "BAD_VALUE"


def test_soft_delete_and_recover_keep_times(env):
    r, _ = env
    before = _times(r, "both")
    assert r.purge_mgr._soft_delete_concept("both")
    assert r.purge_mgr.recover_deleted("both")
    after = _times(r, "both")
    assert (after["c"], after["u"], after["e"]) == (before["c"], before["u"], before["e"])


def test_migration_backfills_times(env):
    r, _ = env
    r.conn.execute("MATCH (c:Concept) SET c.created_at = NULL, c.updated_at = NULL, "
                   "c.effective_at = NULL")
    r.schema_mgr._migrate_v9_to_v10()
    r.schema_mgr._migrate_v9_to_v10()  # idempotent: a re-run after a crash is clean
    assert _times(r, "dated")["e"] == datetime(2024, 1, 2)
    plain = _times(r, "plain")
    assert plain["c"] is not None and plain["e"] == plain["u"]


def test_v9_file_migrates_on_open(tmp_path):
    """A real v9 file: Concept without the time columns, search indexes
    built. Reopening runs the ALTERs and the backfill."""
    from okfgraph.components import embedding as E

    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "old.md").write_text("---\ntitle: old\ntype: note\ndate: 2021-07-08\n---\n\n"
                               "wombat old\n", encoding="utf-8")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(E.LazyRustEncoder, "encode_batch",
                   lambda self, texts, task="Document": [_vec(t, 512) for t in texts])
        mp.setattr(E.LazyRustEncoder, "count_tokens",
                   lambda self, text: len((text or "").split()))
        r = OKFRouter(db_path=str(tmp_path / "kb.db"), bundle_root=str(kb),
                      embedding_dim=512, device="cpu")
        r.import_bundle()
        for col in ("created_at", "updated_at", "effective_at", "content_sig"):
            r.conn.execute(f"ALTER TABLE Concept DROP {col}")
        r.schema_mgr._set_meta("schema_version", 9)
        r.close()
        r = OKFRouter(db_path=str(tmp_path / "kb.db"), bundle_root=str(kb),
                      embedding_dim=512, device="cpu")
        try:
            assert r.schema_mgr._get_meta("schema_version") == 10
            t = _times(r, "old")
            assert t["c"] is not None and t["e"] == datetime(2021, 7, 8)
        finally:
            r.close()


def test_helpers():
    now = datetime(2026, 1, 31)
    assert T.recency_decay(now, now, 30) == 1.0
    assert T.recency_decay(now - timedelta(days=30), now, 30) == pytest.approx(0.5)
    assert T.recency_decay(None, now, 30) == 0.0
    assert T.recency_decay(now + timedelta(days=5), now, 30) == 1.0
    assert T.utc_naive("2024-01-02T03:00:00+02:00") == datetime(2024, 1, 2, 1)
    assert T.utc_naive("not a date") is None
    prev = {"created_at": datetime(2020, 1, 1), "updated_at": datetime(2021, 1, 1),
            "content_sig": T.content_sig({"title": "x"}, "b")}
    same = T.stamp({"title": "x"}, "b", prev, now)
    assert (same["created_at"], same["updated_at"]) == (datetime(2020, 1, 1),
                                                       datetime(2021, 1, 1))
    changed = T.stamp({"title": "y"}, "b", prev, now)
    assert changed["updated_at"] == now and changed["created_at"] == datetime(2020, 1, 1)


def test_thought_is_dated_by_its_created(env):
    r, _ = env
    r.ingest("thoughts", thoughts="Wombats dig burrows at night.",
             topic="notes", concept_id="notes/burrows", title="Burrows")
    cid = "notes/burrows"
    stamped = r.get_by_id(cid).timestamp
    assert stamped is not None and _times(r, cid)["e"] == stamped
    assert abs((T.utc_now() - stamped).total_seconds()) < 120  # UTC, not local


def test_cli_flags_map_to_search_params():
    args = build_parser().parse_args(
        ["search", "wombat", "--recency", "30", "--recency-weight", "0.5"])
    assert _search_filters(args) == {"recency": 30.0, "recency_weight": 0.5}
