# -*- coding: utf-8 -*-
"""External rerank engine — plan-embedder-loader W9.

The rerank half of the loader seam: the same duck-type contract as
:class:`~okfgraph.components.embedding.LazyReranker` (``score_batch``,
``score_long``, ``count_tokens``, ``max_length``, ``model_id``,
``used_cuda``), driven by the standard rerank wire instead of the crate
session:

    POST <endpoint>/v1/rerank  {"model", "query", "documents", "top_n"}
    →  {"results": [{"index": i, "relevance_score": s}]

(0.12.1) TEI-shape tolerance, probe once per session: when the standard
wire is refused, the engine retries the same query against the host
root's ``POST /rerank`` in the TEI text-embeddings-inference shape —
``{"query", "texts", "truncate": true}`` answered by a *bare* list
``[{"index": i, "score": s}]``. Score rows accept ``relevance_score``
(llama.cpp / Jina / Cohere) or ``score`` (some adapters, TEI) in either
shape. Whatever answers first becomes the session dialect; a combined
error naming both attempts is raised when neither does.

One call scores one query against N documents (llama.cpp ``llama-server
--rerank``, TEI, …). ``score_batch`` groups pairs by query and issues one
call per distinct query; every input pair is scored, order preserved.

Token counting (0.12.1): ``rerank_count`` picks the source used for the
short/long split decision.

- ``"heuristic"`` (default): ``max(8, len/4)``, flagged once per process.
- ``"server-tokenize"``: exact — ``POST <host-root>/tokenize`` (llama.cpp
  serves it at the server root) — but *gated*, because the split is a
  threshold check and exactness only matters near the cap:
  a text at most ``(cap-4)/2`` chars can never push a query+doc pair over
  the cap (tokens ≤ code points for BPE/SentencePiece/Unigram, so chars
  are an upper bound) → heuristic value, zero HTTP calls; a text longer
  than ``cap`` chars is already over-cautiously long → upper bound,
  zero calls. Only the ambiguous middle band asks the host, probe once
  first. Any failure there latches the session to ``heuristic-fallback``
  (one loud warning); the search never waits twice on a counted lane.

Session semantics differ from the crate on purpose: the reranker is the
optional half of retrieval, so the external engine **disables itself for
the session on the first failure** (``down=True`` + one loud warning +
:class:`RerankDownError` on the trip) — keyword + vector search keeps
answering, and ``SearchEngine._cross_scores`` answers unranked. No graph
pins: rerank scores are order-only; identity lives in boot logs and the
spec report.
"""
from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "RerankSpec",
    "RerankDownError",
    "ExternalRerankError",
    "ExternalRerankEngine",
    "rerank_engine_from_settings",
]

logger = logging.getLogger("okfgraph.rerank")

#: First ambiguous-band count probes the host with this text (probe once,
#: then exact counts flow without the extra round-trip).
_PROBE_TEXT = ("The quick brown fox jumps over the lazy dog while "
               "retrieval-augmented generation grounds an answer in the "
               "stored evidence.")

_COUNT_MODES = ("heuristic", "server-tokenize")


@dataclass(frozen=True)
class RerankSpec:
    """The declared rerank contract (provenance + advisory ceilings)."""

    family: str                       # declared identity, e.g. "bge-v2-m3"
    model_id: str                     # the wire-side model field
    max_tokens: int                   # per-pair token ceiling (declared)
    endpoint: str
    api_key_from_env: Optional[str]   # name only — the value never lands here
    count: str = "heuristic"          # declared counting source

    def report(self) -> Dict[str, Any]:
        return {
            "backend": "external", "family": self.family,
            "model_id": self.model_id, "endpoint": self.endpoint,
            "max_tokens": self.max_tokens, "windowing": None,
            "count": self.count,
        }


class ExternalRerankError(RuntimeError):
    """The rerank endpoint refused or failed a request, with the body."""


class RerankDownError(RuntimeError):
    """The external rerank lane tripped mid-search; the search answers
    unranked this once, and later searches skip the lane entirely."""


class ExternalRerankEngine:
    """OpenAI-shaped ``/rerank`` client with a LazyReranker-identical API.

    Mirrors ``LazyReranker`` exactly for ``SearchEngine._cross_scores``:
    ``score_batch(pairs)`` (native logits, input order preserved),
    ``score_long(query, doc)``, ``count_tokens(text, cap=None)``,
    ``max_length`` (the declared per-pair ceiling), ``model_id``,
    ``used_cuda`` (always False — the scores live in another process).
    Writes nothing; has no pins.
    """

    def __init__(self, *, endpoint_url: str, model_id: str, family: str,
                 max_tokens: int = 8192, api_key_env: str = "",
                 count: str = "heuristic",
                 timeout: float = 120.0):
        import os
        if not endpoint_url or not str(endpoint_url).strip():
            raise ValueError(
                "rerank backend='external' needs rerank_endpoint_url")
        if not family or not str(family).strip():
            raise ValueError(
                "rerank backend='external' needs rerank_family (the "
                "provenance label, e.g. 'bge-v2-m3')")
        if count not in _COUNT_MODES:
            raise ValueError(
                f"rerank_count must be one of {_COUNT_MODES}, got {count!r}")
        base = str(endpoint_url).rstrip("/")
        if base.endswith("/rerank"):
            base = base[: -len("/rerank")]
        if base.endswith("/v1"):
            base = base[:-3].rstrip("/")
        # Three URLs off the host root:
        #   count_url — POST /tokenize (llama.cpp serves it at the root)
        #   endpoint_url — the standard /v1/rerank wire
        #   tei_url — the TEI /rerank shape, at the root, no /v1
        self.count_url = base + "/tokenize"
        self.endpoint_url = base + "/v1/rerank"
        self.tei_url = base + "/rerank"
        self.spec = RerankSpec(
            family=str(family), model_id=str(model_id),
            max_tokens=int(max_tokens), endpoint=self.endpoint_url,
            api_key_from_env=api_key_env or None, count=str(count),
        )
        if self.spec.max_tokens < 64:
            raise ValueError(
                f"rerank_max_tokens must be >= 64, got {self.spec.max_tokens}")
        self._headers = {"Content-Type": "application/json"}
        if api_key_env:
            key = os.environ.get(api_key_env, "")
            if not key:
                raise ValueError(
                    f"[rerank] api_key_env={api_key_env} is set but the "
                    f"environment variable is empty or missing")
            self._headers["Authorization"] = f"Bearer {key}"
        self.timeout = float(timeout)
        self._target_warned = False
        self._counter_warned = False
        self._down_reason: Optional[str] = None
        self._lock = threading.Lock()
        # the probing latches (set at most once per session, under ._lock)
        self._dialect: Optional[str] = None          # "standard" | "tei"
        self._probe_done = False                     # /tokenize probe
        self._count_eff: str = str(count)            # effective counting source
        self._cprobe_lock = threading.Lock()

    # ── duck-type surface ─────────────────────────────────────────────────
    @property
    def model_id(self) -> str:
        return self.spec.model_id

    @property
    def max_length(self) -> int:
        """Declared per-pair token ceiling (the short/long split point)."""
        return self.spec.max_tokens

    @property
    def used_cuda(self) -> bool:
        return False          # the cross-encoder lives in another process

    @property
    def is_loaded(self) -> bool:
        """Once down, further searches skip the lane entirely."""
        return not self.down

    @property
    def down(self) -> bool:
        """True after the first failed score (rerank disabled for the
        session; keyword + vector search keeps answering)."""
        return self._down_reason is not None

    @property
    def down_reason(self) -> Optional[str]:
        return self._down_reason

    @property
    def counter(self) -> str:
        """The effective counting source: the declared heuristic stays
        heuristic; a latched ``server-tokenize`` reports
        ``heuristic-fallback``."""
        return self._count_eff

    def count_tokens(self, text: str, cap: Optional[int] = None) -> int:
        """Tokens of *text* for the short/long split against *cap*.

        ``"heuristic"`` (default): ``max(8, len/4)``, flagged once per
        process — the split is approximate, never a write-path budget.

        ``"server-tokenize"``: exact from the host's ``POST /tokenize``,
        but gated — the caller passes the effective split ``cap``
        (``min(max_length, window)``, default the declared ceiling) and:

        - ``len(text) <= (cap-4)//2`` → heuristic value, zero calls. For
          BPE/SentencePiece/Unigram vocabularies tokens never exceed
          code points, so chars bound the tokens from above: both query
          and doc below this bar can never lift ``q+doc+4`` over ``cap``,
          whatever the tokenizer actually does.
        - ``len(text) > cap`` → ``len(text)`` (upper bound), zero calls:
          already over-cautiously long.
        - anything in the ambiguous middle band → exact server count,
          probing ``/tokenize`` once first. Any failure latches the
          session to ``heuristic-fallback`` (one loud warning) and every
          later count is the heuristic.
        """
        chars = len(text or "")
        if self.spec.count == "heuristic":
            self._warn_heuristic_once()
            return max(8, chars // 4)
        eff_cap = int(cap) if cap is not None else self.spec.max_tokens
        half = (eff_cap - 4) // 2
        if chars <= half:
            return max(8, chars // 4)
        if chars > eff_cap:
            return chars                       # over-cautiously long
        if self._count_eff == "heuristic-fallback":
            return max(8, chars // 4)          # latched; nothing more to ask
        try:
            return self._server_count(text)
        except Exception as exc:  # noqa: BLE001 — any counting failure degrades
            with self._lock:
                if self._count_eff == "server-tokenize":
                    self._count_eff = "heuristic-fallback"
                    logger.warning(
                        "[rerank] external engine '%s': token counting on "
                        "%s is unavailable (%s); token counts degrade to "
                        "the len/4 heuristic — the short/long split is "
                        "approximate. Set search.rerank_count='heuristic' "
                        "to acknowledge.", self.spec.family,
                        self.count_url, str(exc)[:160])
            return max(8, chars // 4)

    def _warn_heuristic_once(self) -> None:
        if not self._counter_warned:
            with self._lock:
                if not self._counter_warned:
                    self._counter_warned = True
                    logger.warning(
                        "[rerank] external engine '%s' counts tokens "
                        "heuristically (len/4); the short/long split is "
                        "approximate", self.spec.family)

    def _server_count(self, text: str) -> int:
        """One exact count; probes ``/tokenize`` once before the first."""
        tout = min(self.timeout, 10.0)
        if not self._probe_done:
            with self._cprobe_lock:
                if not self._probe_done:
                    self._tokenize_count(_PROBE_TEXT, tout)
                    self._probe_done = True
        return self._tokenize_count(text, tout)

    def _tokenize_count(self, text: str, timeout: float) -> int:
        body = self._request_json(
            self.count_url, {"content": text}, timeout=timeout)
        tokens = body.get("tokens") if isinstance(body, dict) else None
        if not isinstance(tokens, list):
            raise ExternalRerankError(
                f"unexpected /tokenize response shape: {str(body)[:200]}")
        return len(tokens)

    def score_long(self, query: str, doc: str, window=None, stride=None) -> float:
        """Score one pair. ``window``/``stride`` are accepted and ignored
        (the external engine manages its own context; windowing is None
        in the spec report) — loudly, once."""
        if (window is not None or stride is not None) and not self._target_warned:
            with self._lock:
                if not self._target_warned:
                    self._target_warned = True
                    logger.warning(
                        "[rerank] external engine '%s' has no windowing; "
                        "rerank window/stride ignored", self.spec.family)
        return self._score_pairs([(query, doc)])[0]

    def score_batch(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        """Native logits for ``[(query, doc), ...]``, input order preserved.
        Pairs group by query; one ``/rerank`` call scores each group
        (the session dialect decides the wire shape)."""
        return self._score_pairs(list(pairs))

    def spec_report(self) -> Dict[str, Any]:
        out = self.spec.report()
        out["count"] = self.counter
        out["count_declared"] = self.spec.count
        out["dialect"] = self._dialect
        out["down"] = self.down
        out["down_reason"] = self._down_reason
        return out

    # ── the wire ──────────────────────────────────────────────────────────
    def _score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        if not pairs:
            return []
        if self.down:
            raise RerankDownError(
                f"rerank disabled for the session: {self._down_reason}")
        out: List[Optional[float]] = [None] * len(pairs)
        # group pairs by query, preserving first-seen encounter order
        groups: Dict[str, List[int]] = {}
        for i, (q, _d) in enumerate(pairs):
            groups.setdefault(q, []).append(i)
        try:
            for query, idxs in groups.items():
                scores = self._post_rerank(query, [pairs[i][1] for i in idxs])
                for i, s in zip(idxs, scores):
                    out[i] = float(s)
        except Exception as exc:  # noqa: BLE001 — any lane failure trips
            with self._lock:
                if self._down_reason is None:
                    self._down_reason = str(exc)[:200]
                    logger.warning(
                        "[rerank] external engine '%s' failed (%s); rerank "
                        "disabled for this session — keyword + vector "
                        "search still answers", self.spec.family,
                        self._down_reason)
            raise RerankDownError(str(exc)) from exc
        return [float(s) for s in out]  # type: ignore[arg-type]

    def _post_rerank(self, query: str, documents: List[str]) -> List[float]:
        """Score one group on the session dialect, probing once.

        The standard ``/v1/rerank`` wire goes first; if it is refused or
        fails to parse, the same query retries against the host root in
        the TEI ``/rerank`` shape. The first dialect that answers is
        latched for the session; if neither answers, the combined error
        names both attempts and the lane trips down as usual.
        """
        if self._dialect == "standard":
            return self._post_standard(query, documents)
        if self._dialect == "tei":
            return self._post_tei(query, documents)
        try:
            scores = self._post_standard(query, documents)
        except ExternalRerankError as first:
            note = str(first)
        else:
            with self._lock:
                if self._dialect is None:
                    self._dialect = "standard"
            return scores
        try:
            scores = self._post_tei(query, documents)
        except ExternalRerankError as second:
            raise ExternalRerankError(
                f"{note} | TEI fallback at {self.tei_url} also failed: "
                f"{second}") from second
        with self._lock:
            self._dialect = "tei"
        logger.info(
            "[rerank] '%s' answers the TEI /rerank shape (bare list, "
            "'texts'); the standard wire was refused: %s",
            self.spec.family, note[:120])
        return scores

    def _post_standard(self, query: str, documents: List[str]) -> List[float]:
        payload = {
            "model": self.spec.model_id,
            "query": query,
            "documents": list(documents),
            "top_n": len(documents),
        }
        body = self._request_json(self.endpoint_url, payload)
        results = body.get("results") if isinstance(body, dict) else None
        if not isinstance(results, list):
            raise ExternalRerankError(
                f"unexpected rerank response shape: {str(body)[:200]}")
        if len(results) != len(documents):
            raise ExternalRerankError(
                f"expected {len(documents)} scored documents, got {len(results)}")
        return self._cover(results, len(documents), self.endpoint_url)

    def _post_tei(self, query: str, documents: List[str]) -> List[float]:
        payload = {
            "query": query,
            "texts": list(documents),
            "truncate": True,          # the host truncates oversize pairs
        }
        body = self._request_json(self.tei_url, payload)
        # TEI answers a bare list; accept a wrapped dict too for hosts
        # that emulate either shape.
        rows = body if isinstance(body, list) else (
            body.get("results") if isinstance(body, dict) else None)
        if not isinstance(rows, list):
            raise ExternalRerankError(
                f"unexpected TEI rerank response shape: {str(body)[:200]}")
        if len(rows) != len(documents):
            raise ExternalRerankError(
                f"expected {len(documents)} scored documents, got {len(rows)}")
        return self._cover(rows, len(documents), self.tei_url)

    def _request_json(self, url: str, payload: Dict[str, Any],
                      timeout: Optional[float] = None) -> Any:
        """POST *payload* as JSON, answer the parsed body — a dict or a
        list; refusals raise :class:`ExternalRerankError` with the body.
        Carries the session headers (the bearer key when declared)."""
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=dict(self._headers))
        try:
            with urllib.request.urlopen(
                    req, timeout=timeout or 120.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:300]
            except Exception:  # noqa: BLE001
                pass
            raise ExternalRerankError(
                f"{url} refused {exc.code}: {detail or exc}") from exc
        except urllib.error.URLError as exc:
            raise ExternalRerankError(
                f"{url} unreachable: {exc}") from exc

    def _cover(self, rows: List[Any], n_docs: int, where: str) -> List[float]:
        """Rows → full-coverage score list, order of the documents."""
        cover: List[Optional[float]] = [None] * n_docs
        for row in rows:
            if not isinstance(row, dict):
                raise ExternalRerankError(
                    f"{where} row is not an object: {str(row)[:80]}")
            try:
                idx = int(row.get("index"))
            except (TypeError, ValueError) as exc:
                raise ExternalRerankError(
                    f"bad rerank row {str(row)[:80]}") from exc
            if not (0 <= idx < n_docs) or cover[idx] is not None:
                raise ExternalRerankError(
                    f"rerank index coverage broken at {idx}")
            score = row.get("relevance_score")
            if score is None:
                score = row.get("score")
            if score is None:
                raise ExternalRerankError(
                    f"{where} row {str(row)[:80]} carries no score "
                    "(neither relevance_score nor score)")
            try:
                cover[idx] = float(score)
            except (TypeError, ValueError) as exc:
                raise ExternalRerankError(
                    f"bad rerank score in {str(row)[:80]}") from exc
        missing = cover.count(None)
        if missing:
            raise ExternalRerankError(
                f"{where} missing {missing} document score(s)")
        return [float(s) for s in cover]  # type: ignore[arg-type]


def rerank_engine_from_settings(*, backend: str, model_id: str, family: str,
                                endpoint_url: str, api_key_env: str = "",
                                max_tokens: int = 8192,
                                count: str = "heuristic") -> ExternalRerankEngine:
    """Settings-shaped factory (mirrors engine_from_settings for embeds)."""
    if backend != "external":
        raise ValueError(
            f"rerank_engine_from_settings needs backend='external', "
            f"got {backend!r}")
    return ExternalRerankEngine(
        endpoint_url=endpoint_url, model_id=model_id, family=family,
        max_tokens=max_tokens, api_key_env=api_key_env or "", count=count,
    )
