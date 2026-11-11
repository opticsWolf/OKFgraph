# -*- coding: utf-8 -*-
"""External vision engine — the image lane's HTTP client (0.14.0, plan O3).

The text and rerank lanes could already run out of process; images could
not. ``LazyVisionEncoder`` always opened an embroider vision session in
the caller's process, so a host that must not load ONNX (Weave runs
okfgraph that way) could not embed image content at all and fell back to
captions. This module is the missing client: it posts the **original
image bytes** to an ``embroider.serve`` vision host and takes the vectors
back.

Two rules define the lane:

**The host owns the resize.** okfgraph sends the bytes it read, not a
Pillow-resized RGB buffer. ``embroider.prepare_image`` on the host side
is the single resize contract, which is the point of the lane: host and
in-process session must produce the same vectors for the same image, so
``vision_content_hash`` (model + precision + contract + bytes) keeps
matching across a switch, and no graph re-embeds.

**The host must name itself.** There is no declared mode here. A vision
model's vectors are only useful against its *text partner* — image
vectors live in the same ``ImageAsset.embedding`` column and the same
HNSW index as caption vectors — so a host that cannot answer
``GET /v1/models`` with ``engine: embroider``, ``kind: vision``, its
width, precision and ``text_partner`` is refused rather than trusted.
The probe runs when the first image is encoded (the router builds this
engine inside the lazy vision factory), so text-only work never contacts
the vision host.

**One dead host, one error.** A refusal (HTTP 400 for a corrupt image)
raises :class:`ExternalVisionError`, which ``image_assets`` catches to
embed that image's caption instead. An **outage** (connection refused,
timeout) raises ``OKFError('EXTERNAL_UNAVAILABLE')`` and latches the
engine broken: every later call raises the same error without a request.
A host that is down must not turn a thousand images into a thousand
captions.

Nothing here imports ``embroider``: the partner comes from the identity,
never from ``available_models()``.
"""
from __future__ import annotations

import base64
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence

from okfgraph.components._http import (
    UNAVAILABLE,
    auth_headers,
    normalize_endpoint,
    post_json,
    unavailable,
)
from okfgraph.components.external_embedder import probe_identity
from okfgraph.errors import OKFError

logger = logging.getLogger("okfgraph.embedding")

#: Images per request. The host bounds its body (``serve.MAX_BODY``, 32
#: MiB); base64 inflates by 4/3, so the byte budget below is the real
#: limit and this is only the count guard.
MAX_IMAGES_PER_REQUEST = 8

#: Raw-bytes budget per request (base64 + JSON overhead stay under 32 MiB).
MAX_REQUEST_BYTES = 16 * 1024 * 1024


class ExternalVisionError(RuntimeError):
    """The vision host refused one request (its body is in the message).

    A *per-request* failure: the caller may embed that image's caption
    instead and carry on with the rest of the document. Distinct from
    ``OKFError('EXTERNAL_UNAVAILABLE')``, which means the lane itself is
    down and every following image would fail the same way.
    """


def vision_identity(endpoint_url: str, *, api_key_env: str = "",
                    timeout: float = 10.0) -> Optional[Dict[str, Any]]:
    """``GET /v1/models`` for a vision host; None when it is not embroider.

    Reuses the text lane's probe so both lanes agree on what "an
    embroider host" means (a ``data[0]`` row carrying
    ``engine == "embroider"``).
    """
    return probe_identity(endpoint_url, headers=auth_headers(api_key_env),
                          timeout=timeout)


class ExternalVisionEngine:
    """An embroider vision host as the image lane's encoder.

    Mirrors the crate lane's surface where callers touch it: ``model_id``,
    ``dim``, ``precision``, ``space``, ``text_partner``, ``used_cuda``,
    ``is_loaded``, plus ``encode_image_bytes`` (bytes in, vector out —
    the resize happens on the host).
    """

    def __init__(self, *, endpoint_url: str, identity: Dict[str, Any],
                 embedding_dim: int, image_model_id: Optional[str] = None,
                 image_precision: str = "auto", api_key_env: str = "",
                 timeout: float = 60.0,
                 max_images: int = MAX_IMAGES_PER_REQUEST,
                 max_request_bytes: int = MAX_REQUEST_BYTES):
        self.endpoint_url = normalize_endpoint(endpoint_url)
        self.identity = dict(identity)
        self._headers = {"Content-Type": "application/json",
                         **auth_headers(api_key_env)}
        self.timeout = float(timeout)
        self.max_images = max(1, int(max_images))
        self.max_request_bytes = max(1, int(max_request_bytes))
        self.is_loaded = True
        self._broken: Optional[OKFError] = None

        served = str(identity.get("id") or "")
        if identity.get("kind", "") != "vision":
            raise ValueError(
                f"[embedding] {self.endpoint_url} serves a "
                f"'{identity.get('kind')}' model ({served}), not an image "
                f"one: image_endpoint_url needs a vision host "
                f"(embroider.serve --kind vision)"
            )
        if not served:
            raise ValueError(
                f"[embedding] {self.endpoint_url} named no model id; a "
                f"vision host must describe itself (GET /v1/models)")
        if image_model_id and served != image_model_id:
            raise ValueError(
                f"[embedding] {self.endpoint_url} serves {served}, but "
                f"embedding.image_model_id is {image_model_id}: point "
                f"image_endpoint_url at a host for {image_model_id}, or "
                f"drop image_model_id to accept the host's model"
            )
        dim = int(identity.get("dim") or 0)
        if dim != int(embedding_dim):
            raise ValueError(
                f"[embedding] {self.endpoint_url} serves {dim}-wide image "
                f"vectors but the graph is {embedding_dim}-wide: the image "
                f"route has no dimensions parameter, so open the host at the "
                f"graph's width (embroider.serve --dim {embedding_dim})"
            )
        precision = str(identity.get("precision") or "")
        if image_precision != "auto" and precision != image_precision:
            raise ValueError(
                f"[embedding] image_precision={image_precision} but the "
                f"vision host at {self.endpoint_url} landed {precision}: "
                f"set image_precision='auto' or start the host on the "
                f"matching device/precision"
            )
        partner = str(identity.get("text_partner") or "")
        if not partner:
            raise OKFError(
                "IMAGE_PIN_MISMATCH",
                f"the vision host at {self.endpoint_url} names no text "
                f"partner for {served} — image vectors would compare "
                f"against nothing",
                fields={"image_model": served},
                remedy="use mode=text (captions), or host a partnered "
                       "vision model",
            )
        self.model_id = served
        self.dim = dim
        self.precision = precision or "external"
        space = identity.get("space")
        self.space: Optional[str] = space if isinstance(space, str) else None
        self.text_partner = partner
        self.used_cuda = str(identity.get("device") or "") == "cuda"
        logger.info(
            "[embedding] %s is an embroider vision host: %s %s dim=%d "
            "space=%s partner=%s",
            self.endpoint_url, self.model_id, self.precision, self.dim,
            self.space, self.text_partner,
        )

    # ── crate-shape surface ───────────────────────────────────────────────
    @property
    def identity_source(self) -> str:
        return "embroider"

    @property
    def broken(self) -> bool:
        return self._broken is not None

    def spec_report(self) -> Dict[str, Any]:
        return {
            "backend": "external",
            "model_id": self.model_id,
            "endpoint": self.endpoint_url,
            "identity": self.identity_source,
            "kind": self.identity.get("kind"),
            "precision": self.precision,
            "space": self.space,
            "text_partner": self.text_partner,
            "dim": self.dim,
            "device": self.identity.get("device"),
            "engine_version": self.identity.get("engine_version"),
            "contract": self.identity.get("contract"),
            "revision": self.identity.get("revision"),
            "max_images_per_request": self.max_images,
            "request_bytes": self.max_request_bytes,
            "broken": self.broken,
        }

    # ── encoding ──────────────────────────────────────────────────────────
    def encode_image_bytes(self, data: bytes) -> List[float]:
        """Embed one image (original bytes; the host resizes)."""
        return self.encode_images_bytes([data])[0]

    def encode_images_bytes(self, items: Sequence[bytes]) -> List[List[float]]:
        """Embed several images, batched by count and by body size."""
        if self._broken is not None:       # a dead host stays dead
            raise self._broken
        out: List[List[float]] = []
        for batch in self._batches([b or b"" for b in items]):
            out.extend(self._post_images(batch))
        return out

    def _batches(self, items: List[bytes]) -> Iterable[List[bytes]]:
        batch: List[bytes] = []
        size = 0
        for item in items:
            if batch and (len(batch) >= self.max_images
                          or size + len(item) > self.max_request_bytes):
                yield batch
                batch, size = [], 0
            batch.append(item)
            size += len(item)
        if batch:
            yield batch

    def _post_images(self, batch: List[bytes]) -> List[List[float]]:
        payload = {
            "model": self.model_id,
            "image": [base64.b64encode(b).decode("ascii") for b in batch],
        }
        url = f"{self.endpoint_url}/images/embeddings"
        try:
            body = post_json(url, payload, headers=self._headers,
                             timeout=self.timeout,
                             error_cls=ExternalVisionError,
                             outage_remedy=(
                                 "start the vision host, or set "
                                 "embedding.image_backend='embroider'"))
        except OKFError as exc:
            # An outage is a statement about the lane: latch it so the
            # rest of the import fails once, loudly, instead of writing a
            # caption for every remaining image.
            if getattr(exc, "code", "") == UNAVAILABLE:
                self._broken = exc
                raise
            raise
        rows = (body.get("data") if isinstance(body, dict) else None) or []
        if len(rows) != len(batch):
            raise ExternalVisionError(
                f"{url} answered {len(rows)} embeddings for {len(batch)} "
                f"images")
        vectors: List[List[float]] = []
        for row in rows:
            vec = row.get("embedding") if isinstance(row, dict) else None
            if not vec or not isinstance(vec, list):
                raise ExternalVisionError(
                    f"{url} row without an embedding: {str(row)[:120]}")
            if len(vec) != self.dim:
                raise ExternalVisionError(
                    f"{url} returned a {len(vec)}-wide vector, expected "
                    f"{self.dim} ({self.model_id})")
            vectors.append(vec)
        return vectors


def build_external_vision(*, endpoint_url: str, embedding_dim: int,
                          image_model_id: Optional[str] = None,
                          image_precision: str = "auto",
                          api_key_env: str = "",
                          timeout: float = 60.0) -> ExternalVisionEngine:
    """Probe the host, then build the engine (router's external branch).

    Raises ``ValueError`` for a host that cannot serve this graph (the
    router lets it surface as a configuration error) and
    ``ExternalEmbeddingError`` when the endpoint cannot be reached.
    """
    if not endpoint_url.lower().startswith(("http://", "https://")):
        raise ValueError(
            f"[embedding] image_endpoint_url must be an http(s) URL for "
            f"image_backend='external', got '{endpoint_url}'")
    identity = vision_identity(endpoint_url, api_key_env=api_key_env)
    if identity is None:
        raise ValueError(
            f"[embedding] {endpoint_url} did not describe itself as an "
            f"embroider host (GET /v1/models): an external image lane "
            f"needs one, because image vectors are only meaningful against "
            f"the text model the host names — set "
            f"image_backend='embroider', or point image_endpoint_url at "
            f"`python -m embroider.serve --kind vision`"
        )
    return ExternalVisionEngine(
        endpoint_url=endpoint_url, identity=identity,
        embedding_dim=embedding_dim, image_model_id=image_model_id,
        image_precision=image_precision, api_key_env=api_key_env,
        timeout=timeout,
    )


__all__ = [
    "ExternalVisionEngine",
    "ExternalVisionError",
    "MAX_IMAGES_PER_REQUEST",
    "MAX_REQUEST_BYTES",
    "build_external_vision",
    "vision_identity",
    "unavailable",
]
