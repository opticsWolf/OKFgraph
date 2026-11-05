# -*- coding: utf-8 -*-
"""External embedding engines — the loader seam (0.12, plan-embedder-loader W1).

One OKFgraph graph is pinned to one embedding model (``embedding_model`` /
``embedding_precision`` in Meta}. ``backend = 'embroider'`` (the default)
keeps the hardwired Jina-v5-through-ONNX lane. ``backend = 'external'``
swaps the engine for an HTTP embedder that speaks the OpenAI-shaped
``POST /embeddings`` wire the llama.cpp servers, TEI and Ollama pools
already speak — no crates, no ONNX in this process.

**Declared, never guessed.** An external engine carries an
:class:`EngineSpec` — family, per-side task prefixes, pooling identity,
native dim, Matryoshka ladder, token ceiling and where token counts come
from. The router never invents any of it, and refuses a graph whose pin
does not match the spec it is asked to reopen with. Cross-family vectors
are a different vector space and must never mix (the same rule the
precision pin has enforced since 0.10).

**Numerics stay with the engine's owner.** The embedder fails with real
exceptions; hosts that want fail-to-silence degradation wrap it (Silk's
``Embedder.embed`` discipline lives on the host side, not here).
"""
from __future__ import annotations

import json
import math
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Jina-map task names → per-side prefix declarations. The declared table
#: is the *only* prefix source; the router never prepends anything an
#: external engine's spec does not carry (nomic-v1.5's mandatory
#: ``search_query:`` / ``search_document:`` contract is the reference).
DEFAULT_PREFIX_TABLES: Dict[str, Dict[str, str]] = {
    "nomic-v1.5": {
        "Query": "search_query: ",
        "Document": "search_document: ",
    },
}

_HONOURS: List[str] = []  # one-shot warnings for the heuristic counter

#: Declared counting sources (W4). ``server-tokenize`` = the host's own
#: ``POST /tokenize`` for exact counts (llama.cpp serves it at the root;
#: LM Studio / TEI do not serve it and degrade, loudly). ``gguf-meta`` =
#: a chars/token prior read from the model's GGUF header (pure Python,
#: no tokenizer wheel). ``declared`` = the operator's own
#: ``embed_chars_per_token`` constant. ``heuristic`` = ``max(8, len/4)``.
#: Any declared source that fails at probe time lands on
#: ``heuristic-fallback`` — same arithmetic as ``heuristic``, but named
#: for honesty and warned about exactly once.
COUNTER_SOURCES = ("server-tokenize", "gguf-meta", "declared", "heuristic")


@dataclass(frozen=True)
class EngineSpec:
    """Everything a graph pin needs to trust an external engine."""

    family: str                      # declared identity, e.g. "nomic-v1.5"
    model_id: str                    # the graph's ``embedding_model`` pin value
    native_dim: int                  # width the engine returns untruncated
    matryoshka: Optional[Tuple[int, ...]] = None   # ladder, None = native only
    max_tokens: int = 32768          # the engine's true input ceiling
    pooling: str = "mean"            # "mean" | "last" | "cls" — declared
    prefixes: Optional[Dict[str, str]] = None      # per-task prefixes or None
    counter: str = "heuristic"       # a COUNTER_SOURCES value, declared
    counter_gguf: str = ""           # GGUF path for counter="gguf-meta"
    chars_per_token: float = 0.0     # declared value for counter="declared"

    def counting_parameters(self) -> Dict[str, Any]:
        """The counting declaration, for logs and spec reports."""
        params: Dict[str, Any] = {"counter": self.counter}
        if self.counter_gguf:
            params["counter_gguf"] = self.counter_gguf
        if self.chars_per_token:
            params["chars_per_token"] = self.chars_per_token
        return params

    def supports_dim(self, dim: int) -> bool:
        if dim == self.native_dim:
            return True
        return bool(self.matryoshka and dim in self.matryoshka)


def slice_renorm(vec: List[float], dim: int) -> List[float]:
    """Truncate a Matryoshka vector to ``dim`` and renormalise its head.

    Shared by the ``/ops/embed`` op: both lanes may deliver a smaller
    ladder-step vector from the same engine — an MRL-trained embedding
    keeps its geometry under prefix truncation + head renormalisation.
    """
    head = vec[:dim]
    norm = sum(x * x for x in head) ** 0.5
    if norm <= 0:
        return head
    return [x / norm for x in head]


class ExternalEmbeddingError(RuntimeError):
    """The endpoint refused or failed a request, with the response body."""


class ExternalEmbedEngine:
    """OpenAI-shaped ``/embeddings`` client with crate-identical API shape.

    Mirrors :class:`~okfgraph.components.embedding.LazyRustEncoder` for
    every caller: ``encode(text, task)``, ``encode_batch(texts, task)``,
    ``count_tokens(text)`` (aliasing :meth:`count`), plus ``used_cuda``
    (always False — the embeddings live in another process) and ``dim``.
    """

    def __init__(self, *, endpoint_url: str, model_id: str, family: str,
                 embedding_dim: int, native_dim: Optional[int] = None,
                 matryoshka: Optional[Sequence[int]] = None,
                 max_tokens: int = 32768, pooling: str = "mean",
                 prefixes: Optional[Dict[str, str]] = None,
                 counter: str = "heuristic", counter_gguf: str = "",
                 chars_per_token: float = 0.0, api_key_env: str = "",
                 timeout: float = 120.0, batch_size: int = 32):
        import os
        if counter not in COUNTER_SOURCES:
            raise ValueError(
                f"[embedding] embed_counter must be one of "
                f"{COUNTER_SOURCES}, got '{counter}'")
        if counter == "gguf-meta" and not counter_gguf:
            raise ValueError(
                "[embedding] embed_counter='gguf-meta' needs an "
                "embed_counter_gguf path to the model's GGUF file")
        if counter == "declared" and not (
                0.5 <= float(chars_per_token or 0.0) <= 64.0):
            raise ValueError(
                f"[embedding] embed_counter='declared' needs embed_chars_per_token "
                f"within 0.5..64, got {chars_per_token}")
        self.endpoint_url = self._normalize_endpoint(endpoint_url)
        self.endpoint_model = model_id      # the wire-side model field
        self.model_id = model_id            # the pin value (same string v1)
        self.family = family
        self.embedding_dim = int(embedding_dim)
        self.spec = EngineSpec(
            family=family, model_id=model_id,
            native_dim=int(native_dim) if native_dim else int(embedding_dim),
            matryoshka=tuple(int(d) for d in matryoshka) if matryoshka else None,
            max_tokens=int(max_tokens), pooling=pooling,
            prefixes=dict(prefixes) if prefixes else None,
            counter=counter, counter_gguf=counter_gguf,
            chars_per_token=float(chars_per_token or 0.0),
        )
        if not self.spec.supports_dim(self.embedding_dim):
            raise ValueError(
                f"{family}: embedding_dim={self.embedding_dim} is neither "
                f"its native dim {self.spec.native_dim} nor on its declared "
                f"Matryoshka ladder {self.spec.matryoshka}")
        self._headers = {"Content-Type": "application/json"}
        if api_key_env:
            key = os.environ.get(api_key_env, "")
            if not key:
                raise ValueError(
                    f"[embedding] api_key_env={api_key_env} is set but the "
                    f"environment variable is empty or missing")
            self._headers["Authorization"] = f"Bearer {key}"
        self.timeout = float(timeout)
        self.batch_size = max(1, int(batch_size))
        self._renorm_fallback = False   # the server refused `dimensions`
        self._counter_warned = False
        self._counter_mode: str = counter     # effective; may degrade (W4)
        self._counter_ready = False           # the probe ran
        self._counter_failure = ""            # why a probe failed, for the warn
        self._cpt = 0.0                       # measured/declared chars/token
        self._lock = threading.Lock()

    @staticmethod
    def _normalize_endpoint(url: str) -> str:
        """Base endpoint → the exact ``<base>/embeddings`` wire below.

        Accepts ``http://host:port``, ``.../v1`` or ``.../v1/embeddings``;
        llama.cpp and LM Studio/OpenAI servers all mount under ``/v1``.
        """
        base = url.rstrip("/")
        if base.endswith("/embeddings"):
            base = base[: -len("/embeddings")]
        if not base.endswith("/v1"):
            base = base + "/v1"
        return base

    @staticmethod
    def _tokenize_url(normalized: str) -> str:
        """The exact ``/tokenize`` wire (W4): the server's **root**, not
        under ``/v1`` — llama.cpp b12110 serves ``POST /tokenize`` at the
        top level (LM Studio / TEI do not serve it at all and the declared
        counter degrades, loudly, to the heuristic)."""
        base = normalized[:-3] if normalized.endswith("/v1") else normalized
        return base.rstrip("/") + "/tokenize"

    # ── crate-shape surface ───────────────────────────────────────────────
    @property
    def dim(self) -> int:
        return self.embedding_dim

    @property
    def used_cuda(self) -> bool:
        return False       # the engine runs in another process

    @property
    def precision(self) -> str:
        """The numeric identity this engine stores (its owner's business)."""
        return "external"

    @property
    def honours_dimensions(self) -> bool:
        """True until a response came back at native width while a
        Matryoshka dim was requested — i.e. the endpoint ignored the
        ``dimensions`` parameter and the engine sliced client-side."""
        return not self._renorm_fallback

    def encode(self, text: str, task: str = "Document") -> List[float]:
        vec = self._post_embed([text], task)[0]
        return vec

    def encode_batch(self, texts, task: str = "Document") -> List[List[float]]:
        return self._post_embed(list(texts), task)

    # ── the counting ladder (W4) ──────────────────────────────────────────
    def count(self, text: str) -> int:
        """Token count for budgets, from the declared source.

        ``server-tokenize`` asks the host itself (exact, one HTTP round
        per call — the arithmetic price of truth); ``gguf-meta`` and
        ``declared`` divide the character length by a declared prior;
        ``heuristic`` is ``max(8, len/4)``. A declared source that fails
        its once-per-process probe degrades to ``heuristic-fallback``
        with a single loud warning — never silently.
        """
        self._ensure_counter()
        mode = self._counter_mode
        if mode == "server-tokenize":
            return self._count_on_server(text)
        if mode in ("gguf-meta", "declared"):
            return max(1, math.ceil(len(text or "") / self._cpt))
        return max(8, len(text or "") // 4)     # heuristic / fallback

    def count_tokens(self, text: str) -> int:
        """Alias matching ``LazyRustEncoder.count_tokens`` for the router."""
        return self.count(text)

    @property
    def counter(self) -> str:
        """The **effective** counting source: the declared one, or the
        loud ``heuristic-fallback`` degrade. ``/ops/embed`` reports this."""
        self._ensure_counter()
        return self._counter_mode

    def _ensure_counter(self) -> None:
        """Probe the declared source once; degrade loudly on failure."""
        if self._counter_ready:
            return
        with self._lock:
            if self._counter_ready:
                return
            declared = self.spec.counter
            if declared == "server-tokenize":
                ok = self._probe_tokenize()
            elif declared == "gguf-meta":
                ok = self._probe_gguf()
            else:
                ok = True
                if declared == "declared":
                    self._cpt = self.spec.chars_per_token
            if not ok:
                self._counter_mode = "heuristic-fallback"
                self._warn_counter(
                    f"[embedding] embed_counter={declared} is unavailable "
                    f"({self._counter_failure}); token counts degrade to the "
                    f"len/4 heuristic — budgets are approximate. Declare "
                    f"embed_chars_per_token or set embed_counter='heuristic' "
                    f"to acknowledge.")
            elif declared == "heuristic":
                self._warn_counter(
                    "[embedding] external engine '%s' counts tokens "
                    "heuristically (len/4); budgets are approximate"
                    % self.family)
            self._counter_ready = True

    def _warn_counter(self, message: str) -> None:
        import logging
        logging.getLogger("okfgraph.embedding").warning(message)

    def _probe_tokenize(self) -> bool:
        """Does the host answer ``POST /tokenize``? Asked exactly once."""
        url = self._tokenize_url(self.endpoint_url)
        req = urllib.request.Request(
            url,
            data=json.dumps({"content": "okfgraph tokenize probe "
                                         "for exact batch budgets"}).encode("utf-8"),
            headers=dict(self._headers))
        try:
            with urllib.request.urlopen(
                    req, timeout=min(self.timeout, 10.0)) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            tokens = body.get("tokens") if isinstance(body, dict) else None
            if not isinstance(tokens, list):
                raise ValueError("answer carried no token list")
        except Exception as exc:  # noqa: BLE001 — the degrade is the contract
            self._counter_failure = (
                f"{url}: {type(exc).__name__}: {str(exc)[:120]}")
            return False
        return True

    def _count_on_server(self, text: str) -> int:
        """The exact count, as the host's own tokenizer reports it."""
        url = self._tokenize_url(self.endpoint_url)
        req = urllib.request.Request(
            url, data=json.dumps({"content": text or ""}).encode("utf-8"),
            headers=dict(self._headers))
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            tokens = (body.get("tokens") if isinstance(body, dict)
                      else None) or []
            return len(tokens)      # ints, or dicts when with_pieces is set
        except Exception as exc:  # noqa: BLE001
            # Post-probe failure: loud, not a silent heuristic. The probe
            # validated the route; a mid-run loss is a real condition.
            raise ExternalEmbeddingError(
                f"/tokenize failed after the probe validated it "
                f"({url}: {type(exc).__name__}: {str(exc)[:120]})") from exc

    def _probe_gguf(self) -> bool:
        """Can the declared GGUF header yield a chars/token prior?"""
        from okfgraph.components.gguf_meta import (
            GgufCounterError, chars_per_token_from_gguf,
        )
        try:
            self._cpt = chars_per_token_from_gguf(self.spec.counter_gguf)
            return True
        except (OSError, GgufCounterError) as exc:
            self._counter_failure = (
                f"embed_counter_gguf={self.spec.counter_gguf}: "
                f"{type(exc).__name__}: {str(exc)[:160]}")
            return False

    def spec_report(self) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "backend": "external", "family": self.family,
            "model_id": self.model_id, "endpoint": self.endpoint_url,
            "dim": self.embedding_dim, "native_dim": self.spec.native_dim,
            "matryoshka": list(self.spec.matryoshka or ()),
            "max_tokens": self.spec.max_tokens, "pooling": self.spec.pooling,
            "prefixes": self.spec.prefixes,
            **self.spec.counting_parameters(),
            "counter_effective": self.counter,
        }
        if self.spec.counter == "gguf-meta" and self._cpt:
            # the measured prior belongs in the report even though it
            # is engine state, not a spec declaration (W4)
            report["chars_per_token"] = self._cpt
        return report

    # ── the wire ──────────────────────────────────────────────────────────
    def _prefix(self, text: str, task: str) -> str:
        table = self.spec.prefixes
        if not table:
            return text
        prefix = table.get(task)
        if prefix is None and task.lower() in ("query", "retrieval-query",
                                               "searchquery", "question"):
            prefix = table.get("Query")
        if prefix and not text.startswith(prefix):
            return f"{prefix}{text}"
        return text

    def _request(self, payload: Dict[str, Any]) -> Any:
        req = urllib.request.Request(
            f"{self.endpoint_url}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers=dict(self._headers))
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:400]
            except Exception:  # noqa: BLE001
                pass
            raise ExternalEmbeddingError(
                f"{self.endpoint_url}/embeddings refused {exc.code}: "
                f"{detail or exc}") from exc
        except urllib.error.URLError as exc:
            raise ExternalEmbeddingError(
                f"{self.endpoint_url}/embeddings unreachable: {exc}") from exc

    @staticmethod
    def _slice_renorm(vec: List[float], dim: int) -> List[float]:
        return slice_renorm(vec, dim)

    def _post_embed(self, texts: List[str], task: str) -> List[List[float]]:
        if not texts:
            return []
        want_mrl = (self.spec.matryoshka and
                    self.embedding_dim != self.spec.native_dim)
        out: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = [self._prefix(t or "", task)
                           for t in texts[start:start + self.batch_size]]
            payload: Dict[str, Any] = {
                "model": self.endpoint_model, "input": batch_texts,
            }
            if want_mrl and not self._renorm_fallback:
                payload["dimensions"] = self.embedding_dim
            body = self._request(payload)
            rows = (body.get("data") if isinstance(body, dict) else None) or []
            if len(rows) != len(batch_texts):
                raise ExternalEmbeddingError(
                    f"expected {len(batch_texts)} embeddings, got {len(rows)}")
            batch: List[List[float]] = []
            for row in rows:
                vec = row.get("embedding") if isinstance(row, dict) else None
                if not vec or not isinstance(vec, list):
                    raise ExternalEmbeddingError(
                        f"row without an embedding: {str(row)[:120]}")
                batch.append(self._apply_mrl(vec, want_mrl))
            out.extend(batch)
        return out

    def _apply_mrl(self, vec: List[float], want_mrl: bool) -> List[float]:
        if len(vec) == self.embedding_dim:
            return vec
        # A server without ``dimensions`` support returns native width;
        # fall back to client-side slice + renormalise once, loudly.
        if want_mrl and len(vec) == self.spec.native_dim and \
                self.spec.supports_dim(self.embedding_dim):
            with self._lock:
                if not self._renorm_fallback:
                    self._renorm_fallback = True
                    import logging
                    logging.getLogger("okfgraph.embedding").warning(
                        "[embedding] endpoint returned native dim %d; "
                        "slicing to %d client-side (Matryoshka fallback)",
                        len(vec), self.embedding_dim)
            return self._slice_renorm(vec, self.embedding_dim)
        raise ExternalEmbeddingError(
            f"embedding width {len(vec)} != expected {self.embedding_dim} "
            f"(native {self.spec.native_dim})")


def engine_from_settings(*, backend: str, model_id: str, embedding_dim: int,
                         endpoint_url: str = "", api_key_env: str = "",
                         embed_family: str = "", embed_max_tokens: int = 32768,
                         embed_pooling: str = "mean", embed_prefixes: str = "none",
                         embed_counter: str = "heuristic",
                         embed_counter_gguf: str = "",
                         embed_chars_per_token: float = 0.0,
                         **_ignored) -> ExternalEmbedEngine:
    """Build an :class:`ExternalEmbedEngine` from the settings rows.

    ``embed_prefixes`` is the TOML-friendly declaration: ``none``, a
    built-in family name (e.g. ``nomic-v1.5``), or ``Task=prefix|Task=prefix``
    pairs. The built-in table wins only for known families; unknown family
    names without an explicit table are refused (declared, never guessed).
    """
    if backend != "external":
        raise ValueError(f"backend must be 'external', got '{backend}'")
    if not endpoint_url or not endpoint_url.lower().startswith(("http://", "https://")):
        raise ValueError(
            f"[embedding] endpoint_url must be an http(s) URL for backend "
            f"'external', got '{endpoint_url}'")
    if not embed_family or "/" in embed_family:
        raise ValueError(
            f"[embedding] embed_family must be a short declared family name "
            f"(no '/'), got '{embed_family}'")
    if "/" not in model_id:
        raise ValueError(
            f"embedding.model_id must be 'owner/name', got '{model_id}'")

    prefixes: Optional[Dict[str, str]] = None
    if embed_prefixes.startswith("{"):
        raise ValueError(
            "[embedding] embed_prefixes: inline JSON is not a declarations "
            "surface — use 'none', a built-in family name, or "
            "'Task=prefix|Task=prefix' pairs")
    if embed_prefixes in (None, "", "none"):
        prefixes = None
    elif embed_prefixes in DEFAULT_PREFIX_TABLES:
        prefixes = dict(DEFAULT_PREFIX_TABLES[embed_prefixes])
    elif "=" in embed_prefixes:
        prefixes = {}
        for pair in embed_prefixes.split("|"):
            if "=" not in pair:
                raise ValueError(f"embed_prefixes entry without '=': {pair}")
            key, value = pair.split("=", 1)
            prefixes[key.strip()] = value
        _validate_prefix_keys(prefixes)
    else:
        raise ValueError(
            f"[embedding] unknown embed_prefixes '{embed_prefixes}' for "
            f"family '{embed_family}' — declare a table explicitly")

    if embed_pooling not in ("mean", "last", "cls", ""):
        raise ValueError(
            f"[embedding] embed_pooling must be mean/last/cls, got "
            f"'{embed_pooling}'")
    if embedding_dim <= 0 or embedding_dim > 1024:
        raise ValueError(f"embedding_dim out of range: {embedding_dim}")
    if embed_max_tokens < 64:
        raise ValueError(f"embed_max_tokens too small: {embed_max_tokens}")
    if embed_counter not in COUNTER_SOURCES:
        raise ValueError(
            f"[embedding] embed_counter must be one of {COUNTER_SOURCES}, "
            f"got '{embed_counter}'")
    if embed_counter == "gguf-meta" and not embed_counter_gguf:
        raise ValueError(
            "[embedding] embed_counter='gguf-meta' needs embed_counter_gguf, "
            "the path to the model's GGUF file")
    if embed_counter == "declared" and not (
            0.5 <= float(embed_chars_per_token or 0.0) <= 64.0):
        raise ValueError(
            "[embedding] embed_counter='declared' needs embed_chars_per_token "
            f"within 0.5..64, got '{embed_chars_per_token}'")

    native_dim = native_dim_from_family(embed_family, embedding_dim)
    ladder = ladder_from_family(embed_family, embedding_dim)

    return ExternalEmbedEngine(
        endpoint_url=endpoint_url, model_id=model_id, family=embed_family,
        embedding_dim=embedding_dim, native_dim=native_dim,
        matryoshka=ladder, max_tokens=embed_max_tokens,
        pooling=embed_pooling or "mean", prefixes=prefixes,
        counter=embed_counter, counter_gguf=embed_counter_gguf,
        chars_per_token=float(embed_chars_per_token or 0.0),
        api_key_env=api_key_env,
    )


def _validate_prefix_keys(prefixes: Dict[str, str]) -> None:
    allowed = {"Query", "Document"}
    for key in prefixes:
        if key not in allowed:
            raise ValueError(
                f"embed_prefixes task keys must be in {sorted(allowed)}, "
                f"got '{key}'")


def native_dim_from_family(family: str, dim: int) -> int:
    """Native width for declared reference families (W1 keeps the table
    tiny on purpose). 0 = the family has no entry: the dim itself is the
    native width (graph-pinned), which is legal but means the family's
    true ceiling is the host's responsibility."""
    table = {"nomic-v1.5": 768}
    return table.get(family, dim)


def ladder_from_family(family: str, dim: int) -> Optional[List[int]]:
    """Declared Matryoshka ladders for the reference families."""
    table = {"nomic-v1.5": [64, 128, 256, 512, 768]}
    ladder = table.get(family)
    return ladder if ladder and dim in ladder else None
