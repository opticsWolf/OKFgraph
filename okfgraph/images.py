"""Image asset handling for the OKF ingestion pipeline.

This module is intentionally dependency-light: it imports only the standard
library at module load time (``Pillow`` is imported lazily, only when raw image
bytes actually need to be decoded). That keeps the mode-routing / extraction
logic importable and unit-testable without the heavy embedding stack.

Three ingestion modes control how an image becomes a vector in the unified
``ImageAsset`` index:

==========  ============================  ==============================
Mode        Image WITH alt-text           Image WITHOUT alt-text
==========  ============================  ==============================
``text``    text-embed(alt_text)          text-embed(filename + image #)
``optional``text-embed(alt_text)          omni-embed(image bytes)
``omni``    omni-embed(image bytes)       omni-embed(image bytes)
==========  ============================  ==============================

The text-model and omni-model embeddings live in the *same* Matryoshka vector
space (jina-embeddings-v5), so both can be queried from one index without
reindexing. When ``omni`` is requested but the raw bytes are unavailable
(e.g. a remote URL we don't fetch during ingest), planning falls back to the
text path so ingestion never hard-fails on a missing asset.
"""

from __future__ import annotations

import base64
import mimetypes
import re
import urllib.parse
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple


class IngestMode(str, Enum):
    """How images are turned into embeddings during ingestion."""

    TEXT = "text"          # never load omni; alt-text or filename fallback
    OPTIONAL = "optional"  # omni only for images lacking alt-text
    OMNI = "omni"          # omni for every image

    @classmethod
    def coerce(cls, value: "str | IngestMode | None", default: "IngestMode" = None) -> "IngestMode":
        """Parse a user-supplied mode string, tolerantly."""
        if value is None:
            return default or cls.TEXT
        if isinstance(value, cls):
            return value
        key = str(value).strip().lower()
        aliases = {
            "text": cls.TEXT,
            "text-only": cls.TEXT,
            "text_only": cls.TEXT,
            "alt": cls.TEXT,
            "optional": cls.OPTIONAL,
            "hybrid": cls.OPTIONAL,
            "auto": cls.OPTIONAL,
            "omni": cls.OMNI,
            "full": cls.OMNI,
            "multimodal": cls.OMNI,
        }
        if key not in aliases:
            valid = ", ".join(m.value for m in cls)
            raise ValueError(f"Unknown ingest mode {value!r}. Valid modes: {valid}")
        return aliases[key]


class EmbedRoute(str, Enum):
    """Which encoder produces an image asset's embedding."""

    TEXT = "text"   # embed the supplied caption string with the text model
    OMNI = "omni"   # embed the raw image bytes with the omni model


# ``![alt](src "optional title")`` — alt and title are optional.
_MD_IMAGE_RE = re.compile(
    r"""!\[(?P<alt>[^\]]*)\]\(\s*(?P<src><[^>]+>|[^)\s]+)(?:\s+(?P<title>"[^"]*"|'[^']*'))?\s*\)""",
    re.VERBOSE,
)

# Common raster/vector signatures for content sniffing when an extension lies
# or is absent. Keyed by a leading byte signature.
_MAGIC = [
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
]

_ASSET_NAMESPACE = uuid.UUID("6f6b6661-0000-0000-0000-6f6b66617373")  # stable, project-scoped

# Directory (relative to a bundle/file dir) where okf-asset:// bytes are stored
# on disk prior to ingestion, named ``<asset_id>.<ext>``.
ASSET_STORE_DIRNAME = "_assets"


@dataclass
class ExtractedImage:
    """An image reference pulled from a concept body, possibly with bytes loaded."""

    concept_id: str
    index: int                       # 1-based position within the document
    src: str                         # raw markdown link target
    alt_text: Optional[str] = None   # None or "" both mean "no alt-text"
    filename: str = ""
    mime_type: str = "application/octet-stream"
    data: Optional[bytes] = None     # raw bytes, if resolvable
    asset_id: str = ""

    @property
    def has_alt_text(self) -> bool:
        return bool(self.alt_text and self.alt_text.strip())

    @property
    def has_data(self) -> bool:
        return self.data is not None and len(self.data) > 0


def extract_image_refs(body: str) -> List[Tuple[str, str]]:
    """Return ``(alt_text, src)`` pairs for every markdown image, in document order."""
    refs: List[Tuple[str, str]] = []
    for m in _MD_IMAGE_RE.finditer(body or ""):
        alt = (m.group("alt") or "").strip()
        src = (m.group("src") or "").strip()
        if src.startswith("<") and src.endswith(">"):
            src = src[1:-1].strip()
        if src:
            refs.append((alt, src))
    return refs


def asset_id_for(concept_id: str, index: int, src: str) -> str:
    """Deterministic asset id so re-ingesting the same document is idempotent.

    For ``okf-asset://<id>`` links the embedded id is reused verbatim so an
    already-stored asset keeps its identity across round-trips.
    """
    parsed = _parse_okf_asset(src)
    if parsed:
        return parsed
    return f"img_{uuid.uuid5(_ASSET_NAMESPACE, f'{concept_id}:{index}:{src}')}"


def _parse_okf_asset(src: str) -> Optional[str]:
    """Return the asset id if ``src`` is an ``okf-asset://<id>`` URI, else None."""
    if src.startswith("okf-asset://"):
        ident = src[len("okf-asset://"):].strip().strip("/")
        return ident or None
    return None


def sniff_mime(data: Optional[bytes], filename: str) -> str:
    """Best-effort MIME detection: magic bytes first, then filename extension."""
    if data:
        head = data[:16]
        for sig, mime in _MAGIC:
            if head.startswith(sig):
                return mime
        # crude WEBP / AVIF / HEIC checks (RIFF/ISO-BMFF containers)
        if head[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "image/webp"
        if data[4:8] == b"ftyp":
            brand = data[8:12]
            if brand in (b"avif", b"avis"):
                return "image/avif"
            if brand in (b"heic", b"heix", b"heim", b"heis", b"mif1"):
                return "image/heic"
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


def fallback_caption(filename: str, index: int, concept_id: str) -> str:
    """Caption used for the text path when an image has no alt-text.

    Combines the filename with the image's position in the document (the
    ``filename + document_image_number`` strategy), plus the owning concept for
    a little extra signal in the shared text/vector space.
    """
    name = filename or "image"
    parts = [p for p in (concept_id or "").split("/") if p]
    where = parts[-1] if parts else (concept_id or "document")
    return f"{name} (image {index} in {where})"


def filename_from_src(src: str) -> str:
    """Derive a human-ish filename from a markdown image target."""
    okf = _parse_okf_asset(src)
    if okf:
        return okf
    if src.startswith("data:"):
        # data:[<mime>][;base64],<payload> — synthesize a name from the mime
        header = src[5:].split(",", 1)[0]
        mime = header.split(";", 1)[0] or "image/png"
        ext = mimetypes.guess_extension(mime) or ".bin"
        return f"inline{ext}"
    # strip query/fragment, take the last path segment
    cleaned = src.split("#", 1)[0].split("?", 1)[0]
    cleaned = cleaned.replace("\\", "/").rstrip("/")
    name = cleaned.rsplit("/", 1)[-1]
    return urllib.parse.unquote(name) or "image"


def load_image_bytes(
    src: str,
    *,
    search_dirs: List[Path],
    allow_remote: bool = False,
    allowed_domains: Optional[List[str]] = None,
    bundle_root: Optional[Path] = None,
) -> Optional[bytes]:
    """Resolve raw image bytes for a markdown ``src``.

    Handles inline ``data:`` URIs and local files (resolved against
    ``search_dirs`` in order). Remote ``http(s)`` URLs are not fetched unless
    ``allow_remote`` is set. ``okf-asset://<id>`` links resolve from the
    bundle's on-disk asset store (``<dir>/_assets/<id>.<ext>``); if the bytes
    are not on disk the asset is treated as DB-only and None is returned.

    When ``allowed_domains`` is provided, remote URLs are only fetched if the
    domain is in the allowlist (Gap #9a).

    When ``bundle_root`` is provided, local paths are validated to be within
    the bundle root to prevent path traversal attacks (Gap #9b).
    """
    if src.startswith("data:"):
        return _decode_data_uri(src)

    okf_id = _parse_okf_asset(src)
    if okf_id is not None:
        return _load_asset_store_bytes(okf_id, search_dirs)

    scheme = urllib.parse.urlparse(src).scheme.lower()
    if scheme in ("http", "https"):
        if not allow_remote:
            return None
        # Domain allowlist check (Gap #9a)
        if allowed_domains:
            domain = urllib.parse.urlparse(src).hostname
            if domain and not _domain_allowed(domain, allowed_domains):
                return None
        return _fetch_remote(src)
    if scheme and scheme not in ("file",):
        return None  # unknown scheme (mailto:, ftp:, ...) — skip

    # Block file:// URLs — SSRF risk (Gap #9b)
    if scheme == "file":
        return None

    # Local path (relative). Normalise.
    local = urllib.parse.unquote(src)
    candidate = Path(local)
    tried: List[Path] = []
    if candidate.is_absolute():
        tried.append(candidate)
    else:
        for base in search_dirs:
            tried.append((base / candidate))
    for path in tried:
        try:
            # Path traversal check (Gap #9b)
            if bundle_root and not _is_path_within(path, bundle_root):
                continue  # skip paths outside the bundle root
            if path.is_file():
                return path.read_bytes()
        except OSError:
            continue
    return None


def _is_path_within(file_path: Path, root: Path) -> bool:
    """Check if a file path is within the given root directory.

    Prevents path traversal attacks where a malicious markdown file references
    files outside the bundle root (e.g., ``../etc/passwd``).

    Args:
        file_path: The file path to validate.
        root: The root directory that all paths must be within.

    Returns:
        True if the path is within the root, False otherwise.
    """
    try:
        file_path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _load_asset_store_bytes(asset_id: str, search_dirs: List[Path]) -> Optional[bytes]:
    """Resolve ``okf-asset://<asset_id>`` bytes from the bundle's asset store.

    Looks for ``<dir>/_assets/<asset_id>.<ext>`` (and ``<dir>/<asset_id>.<ext>``)
    under each search dir, returning the first match's bytes. Returns None if
    the asset is not present on disk (i.e. it is already DB-only).
    """
    if not asset_id:
        return None
    for base in search_dirs:
        for root in (base / ASSET_STORE_DIRNAME, base):
            try:
                if not root.is_dir():
                    continue
                exact = root / asset_id
                if exact.is_file():
                    return exact.read_bytes()
                for match in sorted(root.glob(f"{asset_id}.*")):
                    if match.is_file():
                        return match.read_bytes()
            except OSError:
                continue
    return None


def _decode_data_uri(src: str) -> Optional[bytes]:
    try:
        header, payload = src[5:].split(",", 1)
    except ValueError:
        return None
    if ";base64" in header.lower():
        try:
            return base64.b64decode(payload)
        except (ValueError, base64.binascii.Error):
            return None
    return urllib.parse.unquote_to_bytes(payload)


def _fetch_remote(src: str) -> Optional[bytes]:
    """Fetch remote bytes only when explicitly allowed (off by default)."""
    try:
        import urllib.request

        with urllib.request.urlopen(src, timeout=10) as resp:  # noqa: S310 (opt-in)
            return resp.read()
    except Exception:
        return None


def _domain_allowed(domain: str, allowed_domains: List[str]) -> bool:
    """Check if a domain is in the allowlist.

    Supports exact matches and wildcard subdomains (*.example.com).
    Blocks private IP ranges and localhost.
    """
    # Block private/internal addresses
    if domain in ("localhost", "0.0.0.0", "127.0.0.1"):
        return False
    # Block private IP ranges
    if domain.startswith(("10.", "192.168.", "172.16.", "172.17.", "172.18.", "172.19.", "172.20.", "172.21.", "172.22.", "172.23.", "172.24.", "172.25.", "172.26.", "172.27.", "172.28.", "172.29.", "172.30.", "172.31.")):
        return False

    for pattern in allowed_domains:
        if pattern.startswith("*."):
            # Wildcard subdomain match
            base = pattern[2:]
            if domain == base or domain.endswith("." + base):
                return True
        else:
            # Exact match
            if domain == pattern:
                return True
    return False


def build_extracted_images(
    concept_id: str,
    body: str,
    *,
    search_dirs: List[Path],
    allow_remote: bool = False,
    allowed_domains: Optional[List[str]] = None,
    bundle_root: Optional[Path] = None,
) -> List[ExtractedImage]:
    """Extract every image ref from ``body`` and resolve bytes/metadata."""
    images: List[ExtractedImage] = []
    for i, (alt, src) in enumerate(extract_image_refs(body), start=1):
        filename = filename_from_src(src)
        data = load_image_bytes(
            src,
            search_dirs=search_dirs,
            allow_remote=allow_remote,
            allowed_domains=allowed_domains,
            bundle_root=bundle_root,
        )
        images.append(
            ExtractedImage(
                concept_id=concept_id,
                index=i,
                src=src,
                alt_text=alt or None,
                filename=filename,
                mime_type=sniff_mime(data, filename),
                data=data,
                asset_id=asset_id_for(concept_id, i, src),
            )
        )
    return images


def plan_embedding(img: ExtractedImage, mode: IngestMode) -> Tuple[EmbedRoute, Optional[str]]:
    """Decide how a single image should be embedded under ``mode``.

    Returns ``(route, caption)``. For ``EmbedRoute.TEXT`` the caption is the
    string to embed with the text model; for ``EmbedRoute.OMNI`` the caption is
    ``None`` (the raw bytes are embedded instead).

    If ``omni`` is selected but no bytes are available, the plan degrades
    gracefully to the text route using the best available caption, so a missing
    or remote asset never aborts ingestion.
    """
    caption = img.alt_text if img.has_alt_text else fallback_caption(
        img.filename, img.index, img.concept_id
    )

    if mode is IngestMode.TEXT:
        return EmbedRoute.TEXT, caption

    if mode is IngestMode.OPTIONAL:
        if img.has_alt_text:
            return EmbedRoute.TEXT, img.alt_text
        if img.has_data:
            return EmbedRoute.OMNI, None
        return EmbedRoute.TEXT, caption  # graceful: no bytes -> use fallback caption

    # IngestMode.OMNI — everything through the omni model when bytes exist
    if img.has_data:
        return EmbedRoute.OMNI, None
    return EmbedRoute.TEXT, caption  # graceful fallback when bytes are missing
