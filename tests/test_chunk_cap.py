"""Post-split chunk cap tests (0.5.1) — no model, no session.

mordant splits purely by block structure with no size limit, so
EmbeddingEngine._split_into_chunks subdivides any block over chunk_size
TOKENS (measured with the tokenizer counter in production) into
exact-tiling continuation pieces ("Type+"). Covered here:

* pure `_cap_chunk_budget`: passthrough, token-driven splitting (a stub
  counter proves splits follow tokens, not words), exact tiling (incl.
  odd whitespace/unicode), byte-offset tiling, suffixing, degenerate
  inputs (wordless, single giant word)
* cold-router integration: cap enforced with a stub counter, dense
  indices, zero-gap offsets, tails chain across prose continuations but
  never touch code pieces
* reconstruct round-trips byte-identical through the REAL
  reconstruct_document (rows inserted directly; no encoding involved)
* pre-cap graphs (no "+" types) take the identical path as before
"""

import pytest

from okfgraph.components.embedding import (
    CONTINUATION_SUFFIX,
    _base_block_type,
    _cap_chunk_budget,
)

WORDS = lambda t: len(t.split())  # noqa: E731 — word fallback, mirrors prod default
CHARS = lambda t: len(t)  # noqa: E731 — stub "tokenizer": 1 char == 1 token


def _chunk(text, block_type="Paragraph", start=0):
    return {
        "parent_doc_id": "d",
        "chunk_text": text,
        "block_type": block_type,
        "start_offset": start,
        "end_offset": start + len(text.encode("utf-8")),
        "chunk_index": 0,
        "heading_context": "",
    }


def _assert_exact_tiling(parent, pieces):
    assert "".join(p["chunk_text"] for p in pieces) == parent["chunk_text"]
    assert pieces[0]["start_offset"] == parent["start_offset"]
    assert pieces[-1]["end_offset"] == parent["end_offset"]
    for a, b in zip(pieces, pieces[1:]):
        assert a["end_offset"] == b["start_offset"]
    raw = parent["chunk_text"].encode("utf-8")
    base = parent["start_offset"]
    for p in pieces:
        assert raw[p["start_offset"] - base:p["end_offset"] - base] == \
            p["chunk_text"].encode("utf-8")


# ---- pure unit -------------------------------------------------------------

def test_under_cap_passes_through_untouched():
    c = _chunk("one two three")
    out = _cap_chunk_budget(c, 512, WORDS)
    assert out == [c] and out[0] is c


def test_wordless_chunk_passes_through():
    c = _chunk("   \n\n  ", block_type="Paragraph")
    assert _cap_chunk_budget(c, 64, WORDS) == [c]


def test_single_giant_word_emitted_alone_but_terminates():
    c = _chunk("a" * 10000)
    out = _cap_chunk_budget(c, 64, WORDS)
    assert len(out) == 1 and out[0]["chunk_text"] == "a" * 10000


def test_splits_follow_tokens_not_words():
    # 200 short words: 200 words but ~890 "tokens" — must split at 512
    # under a token counter, would NOT split under a word counter.
    c = _chunk(" ".join(f"w{i}" for i in range(200)))
    token_out = _cap_chunk_budget(c, 512, CHARS)
    assert len(token_out) == 2
    assert all(CHARS(p["chunk_text"]) <= 512 for p in token_out)
    word_out = _cap_chunk_budget(c, 512, WORDS)
    assert word_out == [c]
    _assert_exact_tiling(c, token_out)


def test_pieces_tile_exactly_with_odd_whitespace():
    text = "  w1  w2\n\nw3\tw4   w5\nw6  "
    c = _chunk("prefix " + text + " suffix", start=10)
    out = _cap_chunk_budget(c, 2, WORDS)
    assert len(out) == 4  # 8 words, cap 2
    assert all(len(p["chunk_text"].split()) <= 2 for p in out)
    _assert_exact_tiling(c, out)


def test_first_piece_keeps_type_continuations_suffixed():
    c = _chunk(" ".join(f"w{i}" for i in range(10)), block_type="CodeBlock")
    out = _cap_chunk_budget(c, 4, WORDS)
    assert [p["block_type"] for p in out] == \
        ["CodeBlock", "CodeBlock+", "CodeBlock+"]


def test_unicode_multibyte_offsets():
    text = " ".join(["wörd—x"] * 10)  # multibyte chars throughout
    c = _chunk(text, start=7)
    out = _cap_chunk_budget(c, 4, WORDS)
    _assert_exact_tiling(c, out)


def test_base_type_helper():
    assert _base_block_type("CodeBlock") == "CodeBlock"
    assert _base_block_type("CodeBlock" + CONTINUATION_SUFFIX) == "CodeBlock"
    assert CONTINUATION_SUFFIX == "+"


# ---- cold-router integration -------------------------------------------------

def _router(tmp_path, name, **kwargs):
    from okfgraph.router import OKFRouter

    return OKFRouter(
        db_path=str(tmp_path / name), bundle_root=str(tmp_path), **kwargs)


def test_cap_enforced_with_dense_indices_and_zero_gap_offsets(tmp_path):
    r = _router(tmp_path, "cap.db", chunk_size=50)
    try:
        words = " ".join(f"w{i}" for i in range(130))
        body = f"# Title\n\n{words}\n"
        chunks = r.embed_engine._split_into_chunks(body, "doc",
                                                   count_tokens=WORDS)
        assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
        assert all(len(c["chunk_text"].split()) <= 50 for c in chunks)
        assert len(chunks) > 2  # heading + several continuations
        assert chunks[0]["block_type"] == "Heading"
        assert chunks[1]["block_type"] == "Paragraph"
        assert all(c["block_type"] == "Paragraph+"
                   for c in chunks[2:])
        # continuation offsets tile the parent byte span
        raw = body.encode("utf-8")
        assert chunks[1]["start_offset"] == raw.index(
            chunks[1]["chunk_text"].encode("utf-8"))
        span_end = chunks[-1]["end_offset"]
        assert b"".join(
            raw[c["start_offset"]:c["end_offset"]] for c in chunks[1:]
        ) == raw[chunks[1]["start_offset"]:span_end]
    finally:
        r.close()


def test_word_fallback_keeps_cold_paths_tokenizer_free(tmp_path):
    r = _router(tmp_path, "cold.db", chunk_size=10)
    try:
        assert r.encoder.is_loaded is False
        body = "# T\n\n" + " ".join(f"w{i}" for i in range(25)) + "\n"
        chunks = r.embed_engine._split_into_chunks(body, "doc")
        assert all(len(c["chunk_text"].split()) <= 10 for c in chunks)
        assert r.encoder.is_loaded is False  # still cold: no tokenizer I/O
    finally:
        r.close()


def test_tails_chain_across_prose_but_never_touch_code(tmp_path):
    r = _router(tmp_path, "tails.db", chunk_size=20, chunk_overlap=5)
    try:
        prose = " ".join(f"p{i}" for i in range(50))
        code = "\n".join(f"line{i} = {i}" for i in range(30))
        body = f"{prose}\n\n```\n{code}\n```\n"
        chunks = r.embed_engine._split_into_chunks(body, "doc",
                                                   count_tokens=WORDS)
        payloads = r.embed_engine._compute_overlap_payloads(chunks)
        by_id = {p["chunk_id"]: p["text"] for p in payloads}
        prose_cont = [c for c in chunks
                      if c["block_type"] == "Paragraph+"]
        assert prose_cont, "expected prose continuations"
        # second prose piece carries the tail of the first (5 words)
        first_words = chunks[0]["chunk_text"].split()
        tail = "  ".join(first_words[-5:])
        assert by_id[f"doc#chunk:{prose_cont[0]['chunk_index']}"].startswith(tail)
        # code pieces embed pure: no prose tail leaks into code
        for c in chunks:
            if _base_block_type(c["block_type"]) == "CodeBlock":
                cid = f"doc#chunk:{c['chunk_index']}"
                assert by_id[cid] == c["chunk_text"]
    finally:
        r.close()


def test_tail_bounded_by_receiver_size(tmp_path):
    r = _router(tmp_path, "bound.db", chunk_overlap=40)
    try:
        big = {"parent_doc_id": "d", "chunk_text": " ".join(f"w{i}" for i in range(100)),
               "block_type": "Paragraph", "start_offset": 0, "end_offset": 1,
               "chunk_index": 0, "heading_context": ""}
        small = {**big, "chunk_text": " ".join(f"s{i}" for i in range(10)),
                 "chunk_index": 1}
        payloads = r.embed_engine._compute_overlap_payloads([big, small])
        tail = payloads[1]["text"].split("\n\n")[0]
        # receiver has 10 words -> tail truncated from 40 to 10
        assert tail == "  ".join(f"w{i}" for i in range(90, 100))
    finally:
        r.close()


def test_tail_untouched_for_full_size_receiver(tmp_path):
    r = _router(tmp_path, "full.db", chunk_overlap=40)
    try:
        a = {"parent_doc_id": "d", "chunk_text": " ".join(f"a{i}" for i in range(100)),
             "block_type": "Paragraph", "start_offset": 0, "end_offset": 1,
             "chunk_index": 0, "heading_context": ""}
        b = {**a, "chunk_text": " ".join(f"b{i}" for i in range(100)),
             "chunk_index": 1}
        payloads = r.embed_engine._compute_overlap_payloads([a, b])
        assert payloads[1]["text"].startswith(
            "  ".join(f"a{i}" for i in range(60, 100)))
    finally:
        r.close()


def test_tail_floor_one_word_never_zero_context(tmp_path):
    r = _router(tmp_path, "floor.db", chunk_overlap=40)
    try:
        big = {"parent_doc_id": "d", "chunk_text": " ".join(f"w{i}" for i in range(100)),
               "block_type": "Paragraph", "start_offset": 0, "end_offset": 1,
               "chunk_index": 0, "heading_context": ""}
        tiny = {**big, "chunk_text": "Alone", "chunk_index": 1}
        payloads = r.embed_engine._compute_overlap_payloads([big, tiny])
        assert payloads[1]["text"].split("\n\n")[0] == "w99"
    finally:
        r.close()


# ---- reconstruct round-trip (real function, direct inserts) -------------------

def _store_chunks(router, doc_id, chunks, dim=512):
    for c in chunks:
        router.conn.execute(
            """CREATE (ch:Chunk {
                id: $id, parent_doc_id: $doc_id, chunk_index: $idx,
                chunk_text: $text, block_type: $bt,
                start_offset: $s, end_offset: $e, embedding: $emb
            })""",
            {"id": f"{doc_id}#chunk:{c['chunk_index']}", "doc_id": doc_id,
             "idx": c["chunk_index"], "text": c["chunk_text"],
             "bt": c["block_type"], "s": c["start_offset"],
             "e": c["end_offset"], "emb": [0.0] * dim},
        )


def test_reconstruct_round_trips_split_blocks_byte_identical(tmp_path):
    r = _router(tmp_path, "recon.db", chunk_size=30)
    try:
        prose = " ".join(f"w{i}" for i in range(100))
        code = "\n".join(f"statement_{i} = compute({i})" for i in range(40))
        items = "- alpha\n- beta\n- gamma"
        quote = "> first line\n> second line"
        # NOTE: no trailing newline — mordant spans never cover a trailing
        # separator (pre-existing "~98% fidelity" behaviour).
        body = f"# Doc\n\n{prose}\n\n```py\n{code}\n```\n\n{items}\n\n{quote}"
        chunks = r.embed_engine._split_into_chunks(body, "doc",
                                                   count_tokens=WORDS)
        assert any(c["block_type"].endswith("+") for c in chunks)
        _store_chunks(r, "doc", chunks)
        assert r.embed_engine.reconstruct_document("doc") == body
    finally:
        r.close()


def test_reconstruct_precap_graph_unchanged(tmp_path):
    r = _router(tmp_path, "compat.db", chunk_size=10000)
    try:
        body = "# T\n\nShort para one.\n\n- a\n- b\n\n> quoted"
        chunks = r.embed_engine._split_into_chunks(body, "doc")
        assert not any(c["block_type"].endswith("+") for c in chunks)
        _store_chunks(r, "doc", chunks)
        assert r.embed_engine.reconstruct_document("doc") == body
    finally:
        r.close()
