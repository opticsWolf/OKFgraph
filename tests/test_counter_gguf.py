# -*- coding: utf-8 -*-
"""W4 counter ladder — the GGUF-header prior and the engine's sources.

Two layers:

* :mod:`okfgraph.components.gguf_meta` parses a synthetic GGUF header
  (magic/version/KV stream written byte-by-byte with struct) and measures
  a chars/token prior from ``tokenizer.ggml.tokens``; a truncated file, a
  non-string array, a special-tokens-only vocab or a missing tokens array
  all refuse loudly.
* The engine consumes it: ``gguf-meta`` and ``declared`` counts divide by
  the prior, ``server-tokenize`` counts exactly (stubbed ``/tokenize`` at
  the server root), and the loud-degrade contract holds — every declared
  source that fails its probe lands on ``heuristic-fallback`` and warns
  exactly once.
"""
from __future__ import annotations

import json
import math
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from okfgraph.components.gguf_meta import (
    GgufCounterError, chars_per_token_from_gguf, _PROBE_TEXT,
)
from okfgraph.components.external_embedder import engine_from_settings


# ── synth GGUF files ─────────────────────────────────────────────────────────

def _string(b, s: str) -> None:
    data = s.encode("utf-8")
    b += struct.pack("<Q", len(data)) + data


def _write_gguf(path, kvs, version=3, truncate=0):
    """``kvs`` = list of (key, kind, value).

    kind 8 = STRING, kind 4 = UINT32, `arr` = (elemtype, str-items) —
    non-string elem types are written as a bare count + zero payload (the
    parser must refuse before reading any of it).
    """
    b = bytearray()
    b += b"GGUF"
    b += struct.pack("<I", version)
    b += struct.pack("<Q", 0)                       # tensor count
    b += struct.pack("<Q", len(kvs))
    for key, kind, value in kvs:
        _string(b, key)
        if kind == 8:
            b += struct.pack("<I", 8)
            _string(b, value)
        elif kind == 4:
            b += struct.pack("<I", 4)
            b += struct.pack("<Q", int(value))
        else:                                       # ("arr", elemtype, items)
            elemtype, items = value
            b += struct.pack("<I", 9)               # value type ARRAY
            b += struct.pack("<I", elemtype)
            b += struct.pack("<Q", len(items))
            if elemtype == 8:
                for item in items:
                    _string(b, item)
            else:
                b += b"\x00" * 8                    # payload the parse must skip
    data = bytes(b)
    if truncate:
        data = data[:-truncate]
    path.write_bytes(data)
    return path


def _vocab(n=1400):
    """A vocab that actually covers the probe: its own unique words (plus
    sentencepiece-marked variants) and BMPE-style subword filler."""
    import re
    seen, uniq = set(), []
    for word in re.findall(r"[A-Za-z][A-Za-z']+", _PROBE_TEXT):
        if word not in seen:
            seen.add(word)
            uniq.append(word)
    pieces = list(uniq) + [f"▁{w}" for w in uniq]
    filler = [f"piece{idx}" for idx in range(max(0, n - len(pieces) - 4))]
    return ["<s>", "</s>", "<unk>", "<pad>"] + pieces + filler


def test_prior_from_a_well_formed_header(tmp_path):
    path = _write_gguf(tmp_path / "model.gguf", [
        ("general.architecture", 8, "nomic-bert"),
        ("tokenizer.ggml.model", 8, "gpt2"),
        ("tokenizer.ggml.tokens", "arr", (8, _vocab())),
    ])
    prior = chars_per_token_from_gguf(str(path))
    assert 1.4 <= prior <= 12.0       # measured on ordinary prose, not 1/4
    assert prior > 1.4                # real coverage, not byte-fallback


def test_normalised_space_markers_make_marked_pieces_match():
    """Sentencepiece pieces arrive prefixed with U+2581; the piece set
    must contain their space-normalised variants so greedy matching can
    fire on real text (the probe never contains the marker)."""
    from okfgraph.components.gguf_meta import (
        _PROBE_TEXT, _chars_per_token_from_vocab, _piece_set,
    )
    import re
    assert " the" in _piece_set(["▁the"])
    specials = ["<s>", "</s>", "<unk>"]
    junk = ["▁zzqx", "▁wgrovf", "▁pmwlk"]            # match nothing
    filler = [f"sub{i}" for i in range(1100)]
    # what a real SPM vocab looks like: most prose words arrive marked
    marked = [f"▁{w}" for w in dict.fromkeys(
        re.findall(r"[A-Za-z][A-Za-z']+", _PROBE_TEXT))]
    baseline = _chars_per_token_from_vocab(specials + junk + filler)
    covered = _chars_per_token_from_vocab(specials + junk + filler + marked)
    assert covered > baseline + 3.0                  # the same pieces merge
    # into words only because the normalised variants match real text;
    # without them everything falls back to byte-per-piece (~1.0).


def test_missing_tokens_array_refuses(tmp_path):
    path = _write_gguf(tmp_path / "no-vocab.gguf", [
        ("general.architecture", 8, "x"),
        ("something.else", 4, 17),
    ])
    with pytest.raises(GgufCounterError, match="no tokenizer.ggml.tokens"):
        chars_per_token_from_gguf(str(path))


def test_bad_magic_refuses(tmp_path):
    path = tmp_path / "junk.gguf"
    path.write_bytes(b"NOPE" + b"\x00" * 40)
    with pytest.raises(GgufCounterError, match="not a GGUF file"):
        chars_per_token_from_gguf(str(path))


def test_truncated_header_refuses(tmp_path):
    path = _write_gguf(tmp_path / "cut.gguf", [
        ("tokenizer.ggml.tokens", "arr", (8, _vocab())),
    ], truncate=64)
    with pytest.raises(GgufCounterError):
        chars_per_token_from_gguf(str(path))


def test_specials_only_vocab_refuses(tmp_path):
    # big enough to pass the vocab-size rail, all specials
    path = _write_gguf(tmp_path / "sp.gguf", [
        ("tokenizer.ggml.tokens", "arr",
         (8, [f"<tag{i}>" for i in range(1100)])),
    ])
    with pytest.raises(GgufCounterError, match="special tokens"):
        chars_per_token_from_gguf(str(path))


def test_non_string_array_refuses(tmp_path):
    path = _write_gguf(tmp_path / "nums.gguf", [
        ("tokenizer.ggml.tokens", "arr", (4, _vocab())),
    ])
    with pytest.raises(GgufCounterError, match="not strings"):
        chars_per_token_from_gguf(str(path))


def test_nested_array_refuses(tmp_path):
    path = _write_gguf(tmp_path / "nested.gguf", [
        ("tokenizer.ggml.tokens", "arr", (9, _vocab())),
    ])
    with pytest.raises(GgufCounterError, match="not strings"):
        chars_per_token_from_gguf(str(path))


# ── the engine's consumption ─────────────────────────────────────────────────

def test_engine_gguf_meta_counts_by_the_prior(tmp_path):
    path = _write_gguf(tmp_path / "model.gguf", [
        ("tokenizer.ggml.tokens", "arr", (8, _vocab())),
    ])
    prior = chars_per_token_from_gguf(str(path))
    eng = engine_from_settings(
        backend="external", model_id="owner/nomic-static",
        embedding_dim=512, endpoint_url="http://127.0.0.1:1",
        embed_family="nomic-v1.5", embed_prefixes="nomic-v1.5",
        embed_counter="gguf-meta", embed_counter_gguf=str(path))
    assert eng.count("x" * 200) == max(1, math.ceil(200 / prior))
    report = eng.spec_report()
    assert report["counter_effective"] == "gguf-meta"
    assert report["counter_gguf"] == str(path)
    assert report["chars_per_token"] == pytest.approx(prior)


def test_engine_declared_counter():
    eng = engine_from_settings(
        backend="external", model_id="owner/nomic-static",
        embedding_dim=512, endpoint_url="http://127.0.0.1:1",
        embed_family="nomic-v1.5", embed_prefixes="nomic-v1.5",
        embed_counter="declared", embed_chars_per_token=3.8)
    assert eng.count("x" * 190) == 50          # ceil(190 / 3.8)
    assert eng.counter == "declared"
    report = eng.spec_report()
    assert report["chars_per_token"] == pytest.approx(3.8)


def test_engine_refuses_counter_forms():
    with pytest.raises(ValueError, match="one of"):
        engine_from_settings(
            backend="external", model_id="owner/x", embedding_dim=512,
            endpoint_url="http://127.0.0.1:1", embed_family="nomic-v1.5",
            embed_counter="tokenizer")          # the pre-W4 spelling is gone
    with pytest.raises(ValueError, match="embed_counter_gguf"):
        engine_from_settings(
            backend="external", model_id="owner/x", embedding_dim=512,
            endpoint_url="http://127.0.0.1:1", embed_family="nomic-v1.5",
            embed_counter="gguf-meta")          # no path
    with pytest.raises(ValueError, match="chars_per_token"):
        engine_from_settings(
            backend="external", model_id="owner/x", embedding_dim=512,
            endpoint_url="http://127.0.0.1:1", embed_family="nomic-v1.5",
            embed_counter="declared")           # no constant
    with pytest.raises(ValueError, match="chars_per_token"):
        engine_from_settings(
            backend="external", model_id="owner/x", embedding_dim=512,
            endpoint_url="http://127.0.0.1:1", embed_family="nomic-v1.5",
            embed_counter="declared", embed_chars_per_token=999.0)


# ── the /tokenize sources ────────────────────────────────────────────────────

class _TokHandler(BaseHTTPRequestHandler):
    """/tokenize at the root (as llama.cpp b12110 serves it) + embeddings."""
    mode = "ints"                               # "ints" | "pieces"
    serve_tokenize = True
    tokenize_hits = []

    def do_POST(self):
        body = json.loads(self.rfile.read(
            int(self.headers.get("Content-Length", 0))))
        if self.path == "/tokenize":
            _TokHandler.tokenize_hits.append(dict(body))
            if not self.serve_tokenize:
                self.send_error(404)
                return
            content = body.get("content", "")
            n = max(1, len(content) // 4)
            if self.mode == "ints":
                payload = {"tokens": list(range(n))}
            else:
                payload = {"tokens": [{"id": i, "piece": "p"}
                                      for i in range(n)]}
            out = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
            return
        if self.path.endswith("/embeddings"):
            dim = int(body.get("dimensions") or 768)
            rows = [{"embedding": _vec(t, dim)} for t in body["input"]]
            out = json.dumps({"data": rows}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
            return
        self.send_error(404)

    def log_message(self, *args):
        pass


def _vec(text: str, dim: int):
    v = [((sum(map(ord, text)) + i * 29) % 1000 - 500) / 500.0
         for i in range(dim)]
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


SERVER = None
PORT = None


def setup_module(module):
    global SERVER, PORT
    _TokHandler.tokenize_hits = []
    _TokHandler.mode = "ints"
    _TokHandler.serve_tokenize = True
    SERVER = ThreadingHTTPServer(("127.0.0.1", 0), _TokHandler)
    PORT = SERVER.server_address[1]
    threading.Thread(target=SERVER.serve_forever, daemon=True).start()


def teardown_module(module):
    if SERVER is not None:
        SERVER.shutdown()


def test_server_tokenize_counts_exactly_at_the_root():
    _TokHandler.serve_tokenize = True
    eng = engine_from_settings(
        backend="external", model_id="owner/nomic-static",
        embedding_dim=768, endpoint_url=f"http://127.0.0.1:{PORT}",
        embed_family="nomic-v1.5", embed_prefixes="nomic-v1.5",
        embed_counter="server-tokenize")
    assert eng.count("hello budgeting world") == 5    # stub: len // 4
    assert _TokHandler.tokenize_hits[-1]["content"] == "hello budgeting world"
    assert eng.spec_report()["counter_effective"] == "server-tokenize"


def test_server_tokenize_accepts_piece_dicts():
    _TokHandler.mode = "pieces"
    try:
        eng = engine_from_settings(
            backend="external", model_id="owner/nomic-static",
            embedding_dim=768, endpoint_url=f"http://127.0.0.1:{PORT}",
            embed_family="nomic-v1.5", embed_prefixes="nomic-v1.5",
            embed_counter="server-tokenize")
        assert eng.count("again we count") == 3    # stub: len // 4
    finally:
        _TokHandler.mode = "ints"


def test_missing_tokenize_degrades_loudly_to_fallback(caplog):
    _TokHandler.serve_tokenize = False
    try:
        eng = engine_from_settings(
            backend="external", model_id="owner/nomic-static",
            embedding_dim=768, endpoint_url=f"http://127.0.0.1:{PORT}",
            embed_family="nomic-v1.5", embed_prefixes="nomic-v1.5",
            embed_counter="server-tokenize")
        with caplog.at_level("WARNING", logger="okfgraph.embedding"):
            assert eng.count("x" * 40) == 10          # max(8, 40 // 4)
            assert eng.counter == "heuristic-fallback"
            hits = [r for r in caplog.records
                    if "degrade to the" in r.message]
            assert len(hits) == 1, "the degrade warns exactly once"
            eng.count("y" * 40)                       # second call
            hits2 = [r for r in caplog.records
                     if "degrade to the" in r.message]
            assert len(hits2) == 1, "no re-warning after the probe"
    finally:
        _TokHandler.serve_tokenize = True


def test_broken_gguf_path_degrades_to_fallback(tmp_path):
    missing = str(tmp_path / "absent.gguf")
    eng = engine_from_settings(
        backend="external", model_id="owner/nomic-static",
        embedding_dim=512, endpoint_url="http://127.0.0.1:1",
        embed_family="nomic-v1.5", embed_prefixes="nomic-v1.5",
        embed_counter="gguf-meta", embed_counter_gguf=missing)
    assert eng.count("x" * 40) == 10
    assert eng.counter == "heuristic-fallback"
