# -*- coding: utf-8 -*-
"""Shared HTTP plumbing for the external lanes (0.14.0).

Three external engines now speak HTTP to a host they do not own: the text
embedder, the reranker, and (0.14.0) the vision encoder. They need the
same four things, so those live here instead of being copied three times:

* endpoint normalisation (``http://h:9`` / ``http://h:9/v1`` /
  ``.../v1/embeddings`` all mean the same base);
* ``<base>/tokenize`` — the counting route lives at the server *root*,
  not under ``/v1``;
* the bearer header declared by ``api_key_env``;
* a JSON request that tells a **refusal** apart from an **outage**.

That last distinction is the reason this module exists. A refusal is a
statement about one request (a 400 for a corrupt image, a 422 for a text
over the ceiling): the caller may degrade per item and carry on. An
outage is a statement about the lane (connection refused, DNS failure,
timeout): every subsequent request will fail the same way, so it must be
one typed error naming the endpoint — never a silent per-item degrade
that turns a dead host into a hundred captions.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from okfgraph.errors import OKFError

#: The code an outage raises. Typed since 0.14.0: the text, rerank and
#: vision lanes all surface it, so a harness can tell "the host is down"
#: from "this one request was bad" without string matching.
UNAVAILABLE = "EXTERNAL_UNAVAILABLE"


def auth_headers(api_key_env: str) -> Dict[str, str]:
    """The bearer header for *api_key_env*, or none when it is unset."""
    import os
    if not api_key_env:
        return {}
    key = os.environ.get(api_key_env, "")
    if not key:
        raise ValueError(
            f"[embedding] api_key_env={api_key_env} is set but the "
            f"environment variable is empty or missing")
    return {"Authorization": f"Bearer {key}"}


def normalize_endpoint(url: str) -> str:
    """Base endpoint → the ``<base>/v1`` every host mounts its API under.

    Accepts ``http://host:port``, ``.../v1`` or ``.../v1/embeddings``;
    llama.cpp, LM Studio/OpenAI and embroider all mount under ``/v1``.
    """
    base = (url or "").rstrip("/")
    if base.endswith("/embeddings"):
        base = base[: -len("/embeddings")]
    if base.endswith("/rerank"):
        base = base[: -len("/rerank")]
    if base.endswith("/images/embeddings"):
        base = base[: -len("/images/embeddings")]
    if not base.endswith("/v1"):
        base = base + "/v1"
    return base


def tokenize_url(normalized: str) -> str:
    """The exact ``/tokenize`` wire: the server's **root**, not ``/v1``.

    llama.cpp serves ``POST /tokenize`` at the top level; embroider's
    ``serve`` does the same. TEI and LM Studio do not serve it at all, and
    the declared counter degrades loudly there.
    """
    base = normalized[:-3] if normalized.endswith("/v1") else normalized
    return base.rstrip("/") + "/tokenize"


def unavailable(url: str, exc: BaseException, *, remedy: str = "") -> OKFError:
    """A typed outage error naming the endpoint (never per-item)."""
    return OKFError(
        UNAVAILABLE,
        f"{url} is unreachable: {type(exc).__name__}: {str(exc)[:160]}",
        fields={"endpoint": url},
        remedy=remedy or ("start the host, or drop the external lane for "
                          "this one so the in-process lane is used"),
    )


def post_json(
    url: str,
    payload: Any,
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 120.0,
    error_cls: type = RuntimeError,
    outage_remedy: str = "",
) -> Any:
    """POST *payload* as JSON and answer the parsed body.

    A refusal raises ``error_cls`` with the response body (the caller may
    degrade per item); an outage raises :class:`OKFError` with
    ``EXTERNAL_UNAVAILABLE`` (the caller must not degrade per item).
    """
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=dict(headers or {"Content-Type": "application/json"}),
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:400]
        except Exception:  # noqa: BLE001
            pass
        raise error_cls(
            f"{url} refused {exc.code}: {detail or exc}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise unavailable(url, exc, remedy=outage_remedy) from exc


def get_json(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 10.0,
    error_cls: type = RuntimeError,
) -> Any:
    """GET *url* and answer the parsed body (refusals → ``error_cls``)."""
    req = urllib.request.Request(url, headers=dict(headers or {}))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise error_cls(f"{url} refused {exc.code}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise unavailable(url, exc) from exc


__all__ = [
    "UNAVAILABLE",
    "auth_headers",
    "normalize_endpoint",
    "tokenize_url",
    "unavailable",
    "post_json",
    "get_json",
]
