# -*- coding: utf-8 -*-
"""A token-count prior read straight from a GGUF file header (W4).

The counting ladder for external engines wants an option between "the
server tokenises exactly" (``embed_counter='server-tokenize'``) and the
``len/4`` heuristic. Most users possess the model's GGUF — their server
loads it — so the header itself can yield a measured prior for how many
characters a token of that vocabulary costs, with zero compiled
dependencies: no HF ``tokenizers`` wheel ever enters the graph (the 0.12
ruling).

How the prior is derived
------------------------
The GGUF header's metadata carries ``tokenizer.ggml.tokens`` — the whole
vocabulary as plain strings. From it we build a piece set (normalising
the sentencepiece ``U+2581`` and GPT-2 ``U+0120`` space markers to real
spaces so both tokenizer styles match ordinary text) and then greedily
longest-match a fixed English probe against that set. The probe's
``chars / pieces`` ratio is the prior: a *measured* estimate on ordinary
prose for this vocabulary, not a magic constant. It is deliberately not
the exact tokenizer — budgeting only needs to be close and lean safe:
batches pack sub-``max_tokens`` per round, and the fence that really
matters (a single text over the ceiling) over-counts by the same margin
the heuristic already does.

Errors raise :class:`GgufCounterError` so the engine can degrade to the
heuristic **loudly, once** — the declared prior failing must never fade
into silent approximation.
"""
from __future__ import annotations

import struct
from typing import BinaryIO, List, Set

#: Metadata value types (GGUF v2/v3 layout).
_UINT8, _INT8, _UINT16, _INT16 = 0, 1, 2, 3
_UINT32, _INT32, _FLOAT32, _BOOL, _STRING, _ARRAY = 4, 5, 6, 7, 8, 9
_UINT64, _INT64, _FLOAT64 = 10, 11, 12
_TYPE_SIZE = {_UINT8: 1, _INT8: 1, _UINT16: 2, _INT16: 2,
              _UINT32: 4, _INT32: 4, _FLOAT32: 4, _BOOL: 1,
              _UINT64: 8, _INT64: 8, _FLOAT64: 8}

#: Sanity rails for a meaningful prior.
_MIN_VOCAB = 1000
_MAX_KV_PAIRS = 4096
_MAX_STRING = 1 << 22          # 4 MiB — no single header string is larger
_MAX_TOKEN_STR = 48            # pieces longer are junk for matching
_MAX_KV_SCAN = 4_000_000       # hard ceiling while skipping vocab arrays
_PROBE_CHARS_PER_TOKEN_RANGE = (1.0, 64.0)


class GgufCounterError(ValueError):
    """The GGUF header could not yield a usable token-count prior."""


# Ordinary English with numbers, punctuation and a touch of jargon — the
# shape the counting actually serves. Repeated so the mean is stable.
_PROBE_TEXT = (
    "Retrieval-augmented generation grounds an answer in the evidence a "
    "knowledge graph stores: every concept, title and chunk of text is "
    "mapped into one vector space, where cosine similarity finds the "
    "nearest neighbours. In 2024 open embedding models reached 768 or "
    "1024 dimensions, with matryoshka truncation trading a little "
    "accuracy for much smaller vectors. The declared contract matters — "
    "prefixes such as search_query: and search_document: flavour each "
    "side of retrieval, mean pooling keeps long passages stable, and a "
    "careful batch budget keeps memory flat while encoding 4,096 rows. "
) * 24


def _read_u(fh: BinaryIO, fmt: str) -> int:
    size = struct.calcsize(fmt)
    data = fh.read(size)
    if len(data) < size:
        raise GgufCounterError("truncated GGUF header")
    return struct.unpack(fmt, data)[0]


def _read_string(fh: BinaryIO) -> str:
    length = _read_u(fh, "<Q")
    if length > _MAX_STRING:
        raise GgufCounterError(f"header string too long: {length}")
    data = fh.read(length)
    if len(data) < length:
        raise GgufCounterError("truncated GGUF header string")
    return data.decode("utf-8", "replace")


def _skip_value(fh: BinaryIO, vtype: int) -> None:
    """Consume one metadata value without keeping it."""
    if vtype == _STRING:
        _read_string(fh)
        return
    if vtype == _ARRAY:
        elem = _read_u(fh, "<I")
        count = _read_u(fh, "<Q")
        if elem == _ARRAY:
            raise GgufCounterError("nested arrays are not supported")
        if elem == _STRING:
            for _ in range(min(count, _MAX_KV_SCAN)):
                _read_string(fh)
            return
        size = _TYPE_SIZE.get(elem)
        if size is None:
            raise GgufCounterError(f"unknown GGUF value type {elem}")
        fh.seek(count * size, 1)
        return
    size = _TYPE_SIZE.get(vtype)
    if size is None:
        raise GgufCounterError(f"unknown GGUF value type {vtype}")
    fh.seek(size, 1)



def _read_token_array(fh: BinaryIO) -> List[str]:
    elem = _read_u(fh, "<I")
    count = _read_u(fh, "<Q")
    if elem != _STRING:
        raise GgufCounterError(
            f"tokenizer.ggml.tokens is a type-{elem} array, not strings")
    if count < _MIN_VOCAB or count > _MAX_KV_SCAN:
        raise GgufCounterError(f"impossible vocab size: {count}")
    return [_read_string(fh) for _ in range(count)]


def _piece_set(tokens: List[str]) -> Set[str]:
    """Matchable pieces: space markers normalised, specials dropped."""
    pieces: Set[str] = set()
    for raw in tokens:
        if not raw or len(raw) > _MAX_TOKEN_STR:
            continue
        if raw[0] == "<" and raw[-1] == ">":
            continue                      # <s>, </s>, <unk>, imeo specials
        pieces.add(raw)
        if "\u2581" in raw:              # sentencepiece space marker
            pieces.add(raw.replace("\u2581", " "))
        if "\u0120" in raw:              # GPT-2 byte space marker
            pieces.add(raw.replace("\u0120", " "))
    return pieces


def _chars_per_token_from_vocab(tokens: List[str]) -> float:
    specials = sum(1 for t in tokens if t[:1] == "<" and t[-1:] == ">")
    if specials * 3 >= len(tokens):
        raise GgufCounterError(
            f"vocab looks like special tokens only ({specials}/{len(tokens)})")
    pieces = _piece_set(tokens)
    if not pieces:
        raise GgufCounterError("no usable pieces in the vocabulary")
    max_len = min(max(len(p) for p in pieces), _MAX_TOKEN_STR)
    chars = pieces_n = 0
    text = _PROBE_TEXT
    i, end = 0, len(text)
    while i < end:
        for width in range(min(max_len, end - i), 0, -1):
            piece = text[i:i + width]
            if piece in pieces:
                chars += width
                pieces_n += 1
                i += width
                break
        else:
            chars += 1                    # an unmatched byte is its own piece
            pieces_n += 1
            i += 1
    if pieces_n == 0:
        raise GgufCounterError("the probe tokenised to nothing")
    prior = chars / pieces_n
    lo, hi = _PROBE_CHARS_PER_TOKEN_RANGE
    if not lo <= prior <= hi:
        raise GgufCounterError(f"implausible prior: {prior:.2f} chars/token")
    return prior


def chars_per_token_from_gguf(path: str) -> float:
    """Read *path*'s GGUF header and return a measured chars/token prior."""
    with open(path, "rb") as fh:
        if fh.read(4) != b"GGUF":
            raise GgufCounterError("not a GGUF file (no GGUF magic)")
        version = _read_u(fh, "<I")
        if version < 2:
            raise GgufCounterError(f"GGUF v{version} unsupported (need v2+)")
        _read_u(fh, "<Q")                 # tensor count — irrelevant here
        kv_count = _read_u(fh, "<Q")
        seen = 0
        while seen < min(kv_count, _MAX_KV_PAIRS):
            key = _read_string(fh)
            vtype = _read_u(fh, "<I")
            seen += 1
            if key == "tokenizer.ggml.tokens":
                return _chars_per_token_from_vocab(_read_token_array(fh))
            _skip_value(fh, vtype)
    raise GgufCounterError(
        f"no tokenizer.ggml.tokens array in {path}'s header "
        f"({_MAX_KV_PAIRS}-pair scan limit)")

__all__ = ["GgufCounterError", "chars_per_token_from_gguf"]
