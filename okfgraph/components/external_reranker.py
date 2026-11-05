# -*- coding: utf-8 -*-
"""External rerank engine — plan-embedder-loader W9.

The rerank half of the loader seam: the same duck-type contract as
:class:`~okfgraph.components.embedding.LazyReranker` (``score_batch``,
``score_long``, ``count_tokens``, ``max_length``, ``model_id``,
``used_cuda``), driven by the standard rerank wire instead of the crate
session:

    POST <endpoint>/v1/rerank  {"model", "query", "documents", "top_n"}
    →  {"results": [{"index": i, "relevance_score": s}]}

One call scores one query against N documents (llama.cpp ``llama-server
--rerank``, TEI, …). ``score_batch`` groups pairs by query and issues one
call per distinct query; every input pair is scored, order preserved.

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


@dataclass(frozen=True)
class RerankSpec:
    """The declared rerank contract (provenance + advisory ceilings)."""

    family: str                       # declared identity, e.g. "bge-v2-m3"
    model_id: str                     # the wire-side model field
    max_tokens: int                   # per-pair token ceiling (declared)
    endpoint: str
    api_key_from_env: Optional[str]   # name only — the value never lands here

    def report(self) -> Dict[str, Any]:
        return {
            "backend": "external", "family": self.family,
            "model_id": self.model_id, "endpoint": self.endpoint,
            "max_tokens": self.max_tokens, "windowing": None,
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
    ``score_long(query, doc)``, ``count_tokens(text)``, ``max_length`` (the
    declared per-pair ceiling), ``model_id``, ``used_cuda`` (always False —
    the scores live in another process). Writes nothing; has no pins.
    """

    def __init__(self, *, endpoint_url: str, model_id: str, family: str,
                 max_tokens: int = 8192, api_key_env: str = "",
                 timeout: float = 120.0):
        import os
        if not endpoint_url or not str(endpoint_url).strip():
            raise ValueError(
                "rerank backend='external' needs rerank_endpoint_url")
        if not family or not str(family).strip():
            raise ValueError(
                "rerank backend='external' needs rerank_family (the "
                "provenance label, e.g. 'bge-v2-m3')")
        base = str(endpoint_url).rstrip("/")
        if base.endswith("/rerank"):
            base = base[: -len("/rerank")]
        if not base.endswith("/v1"):
            base = base + "/v1"
        self.endpoint_url = base + "/rerank"
        self.spec = RerankSpec(
            family=str(family), model_id=str(model_id),
            max_tokens=int(max_tokens), endpoint=self.endpoint_url,
            api_key_from_env=api_key_env or None,
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

    def count_tokens(self, text: str) -> int:
        """Heuristic token count (len/4, floor 8), flagged once per
        process — used only for the short/long split decision, never for
        write-path budgets."""
        if not self._counter_warned:
            with self._lock:
                if not self._counter_warned:
                    self._counter_warned = True
                    import logging
                    logging.getLogger("okfgraph.rerank").warning(
                        "[rerank] external engine '%s' counts tokens "
                        "heuristically (len/4); the short/long split is "
                        "approximate", self.spec.family)
        return max(8, len(text or "") // 4)

    def score_long(self, query: str, doc: str, window=None, stride=None) -> float:
        """Score one pair. ``window``/``stride`` are accepted and ignored
        (the external engine manages its own context; windowing is None
        in the spec report) — loudly, once."""
        if (window is not None or stride is not None) and not self._target_warned:
            with self._lock:
                if not self._target_warned:
                    self._target_warned = True
                    import logging
                    logging.getLogger("okfgraph.rerank").warning(
                        "[rerank] external engine '%s' has no windowing; "
                        "rerank window/stride ignored", self.spec.family)
        return self._score_pairs([(query, doc)])[0]

    def score_batch(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        """Native logits for ``[(query, doc), ...]``, input order preserved.
        Pairs group by query; one `/v1/rerank` call scores each group."""
        return self._score_pairs(list(pairs))

    def spec_report(self) -> Dict[str, Any]:
        out = self.spec.report()
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
                    import logging
                    logging.getLogger("okfgraph.rerank").warning(
                        "[rerank] external engine '%s' failed (%s); rerank "
                        "disabled for this session — keyword + vector "
                        "search still answers", self.spec.family,
                        self._down_reason)
            raise RerankDownError(str(exc)) from exc
        return [float(s) for s in out]  # type: ignore[arg-type]

    def _post_rerank(self, query: str, documents: List[str]) -> List[float]:
        payload = {
            "model": self.spec.model_id,
            "query": query,
            "documents": list(documents),
            "top_n": len(documents),
        }
        req = urllib.request.Request(
            self.endpoint_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=dict(self._headers))
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:300]
            except Exception:  # noqa: BLE001
                pass
            raise ExternalRerankError(
                f"{self.endpoint_url} refused {exc.code}: "
                f"{detail or exc}") from exc
        except urllib.error.URLError as exc:
            raise ExternalRerankError(
                f"{self.endpoint_url} unreachable: {exc}") from exc
        results = body.get("results") if isinstance(body, dict) else None
        if not isinstance(results, list):
            raise ExternalRerankError(
                f"unexpected rerank response shape: {str(body)[:200]}")
        if len(results) != len(documents):
            raise ExternalRerankError(
                f"expected {len(documents)} scored documents, got {len(results)}")
        cover = [None] * len(documents)
        for row in results:
            try:
                idx = int(row.get("index"))
                score = float(row.get("relevance_score"))
            except (TypeError, ValueError, AttributeError) as exc:
                raise ExternalRerankError(
                    f"bad rerank row {str(row)[:80]}") from exc
            if not (0 <= idx < len(documents)) or cover[idx] is not None:
                raise ExternalRerankError(
                    f"rerank index coverage broken at {idx}")
            cover[idx] = score
        return [float(s) for s in cover]  # type: ignore[arg-type]


def rerank_engine_from_settings(*, backend: str, model_id: str, family: str,
                                endpoint_url: str, api_key_env: str = "",
                                max_tokens: int = 8192) -> ExternalRerankEngine:
    """Settings-shaped factory (mirrors engine_from_settings for embeds)."""
    if backend != "external":
        raise ValueError(
            f"rerank_engine_from_settings needs backend='external', "
            f"got {backend!r}")
    return ExternalRerankEngine(
        endpoint_url=endpoint_url, model_id=model_id, family=family,
        max_tokens=max_tokens, api_key_env=api_key_env or "",
    )
