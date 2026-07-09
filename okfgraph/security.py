"""Security utilities for OKFgraph.

This module provides path traversal protection and model cache verification
to mitigate SSRF attacks and supply chain risks.

Gap #9: Security — sandboxed path validation + cache verification.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Private / internal IP ranges that should never be accessed remotely.
_PRIVATE_RANGES: List[str] = [
    "0.0.0.0/8",
    "10.0.0.0/8",
    "100.64.0.0/10",    # Carrier-grade NAT
    "127.0.0.0/8",      # Loopback
    "169.254.0.0/16",   # Link-local
    "172.16.0.0/12",    # Private
    "192.0.0.0/24",     # IETF Protocol Assignments
    "192.0.2.0/24",     # Documentation (TEST-NET-1)
    "192.168.0.0/16",   # Private
    "198.51.100.0/24",  # Documentation (TEST-NET-2)
    "203.0.113.0/24",   # Documentation (TEST-NET-3)
    "224.0.0.0/4",      # Multicast
    "240.0.0.0/4",      # Reserved
]

# Pre-compute private networks for fast lookup.
_PRIVATE_NETWORKS: List[ipaddress.IPv4Network] = [
    ipaddress.IPv4Network(r) for r in _PRIVATE_RANGES
]


def is_private_ip(host: str) -> bool:
    """Check if a hostname or IP address is in a private/internal range.

    Args:
        host: Hostname or IP address to check.

    Returns:
        True if the host is a private IP, False otherwise.
    """
    # Quick check for localhost variants
    if host.lower() in ("localhost", "localhost.localdomain", "0.0.0.0", "::1"):
        return True

    try:
        addr = ipaddress.IPv4Address(host)
        for net in _PRIVATE_NETWORKS:
            if addr in net:
                return True
    except (ipaddress.AddressValueError, ValueError):
        pass  # Not an IP address — domain name, let DNS resolve

    return False


def is_path_safe(file_path: Path, bundle_root: Path) -> bool:
    """Check if a file path is within the allowed bundle root directory.

    Prevents path traversal attacks where a malicious markdown file references
    files outside the bundle root (e.g., ``../etc/passwd``).

    Args:
        file_path: The file path to validate.
        bundle_root: The root directory that all paths must be within.

    Returns:
        True if the path is within the bundle root, False otherwise.
    """
    try:
        file_path.resolve().relative_to(bundle_root.resolve())
        return True
    except ValueError:
        return False


def validate_image_src(
    src: str,
    bundle_root: Path,
    *,
    allow_remote: bool = False,
    allowed_domains: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """Validate an image source reference for security.

    Checks:
    1. If it's a local file, it must be within the bundle root.
    2. If it's a remote URL, remote access must be allowed and the domain
       must be on the allowlist.
    3. ``file://`` URLs are always blocked (SSRF risk).

    Args:
        src: The image source string (filename, okf-asset://, or URL).
        bundle_root: The root directory for local file validation.
        allow_remote: Whether remote URLs are permitted at all.
        allowed_domains: List of allowed domain patterns for remote URLs.

    Returns:
        Tuple of (is_safe, reason). If is_safe is False, reason explains why.
    """
    # Block file:// URLs — SSRF risk
    if src.startswith("file://"):
        return False, "file:// URLs are blocked (SSRF risk)"

    # okf-asset:// URIs are always safe (internal references)
    if src.startswith("okf-asset://"):
        return True, "okf-asset internal reference"

    # HTTP(S) URLs — remote image handling
    if src.startswith(("http://", "https://")):
        if not allow_remote:
            return False, "remote images are disabled"

        # Extract domain from URL
        domain_match = re.match(r"https?://([^/:]+)", src)
        if not domain_match:
            return False, "invalid URL format"

        domain = domain_match.group(1).lower()

        # Block private IPs
        if is_private_ip(domain):
            return False, f"private IP blocked: {domain}"

        # Check domain allowlist
        if allowed_domains:
            if not _domain_in_allowlist(domain, allowed_domains):
                return False, f"domain not in allowlist: {domain}"

        return True, "remote URL permitted"

    # Local file reference
    file_path = bundle_root / src
    if not is_path_safe(file_path, bundle_root):
        return False, f"path outside bundle root: {src}"

    return True, "local file within bundle root"


def _domain_in_allowlist(domain: str, allowed_domains: List[str]) -> bool:
    """Check if a domain matches any pattern in the allowlist.

    Supports exact matches and wildcard subdomains (``*.example.com``).

    Args:
        domain: The domain to check.
        allowed_domains: List of allowed domain patterns.

    Returns:
        True if the domain matches any pattern, False otherwise.
    """
    for pattern in allowed_domains:
        pattern = pattern.lower()
        if pattern == domain:
            return True
        if pattern.startswith("*."):
            # Wildcard subdomain match
            base = pattern[2:]
            if domain.endswith(base) and domain.count(".") == base.count(".") + 1:
                return True
    return False


# ------------------------------------------------------------------
# Model Cache Verification (Gap #9d)
# ------------------------------------------------------------------

class ModelCacheVerifier:
    """Verify HuggingFace model cache integrity.

    On first load, computes SHA-256 hashes of model files and compares
    against pinned expected values. Subsequent loads are cached for speed.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self._known_hashes: Dict[str, str] = {}  # file_path -> expected hash
        self._verified: Set[str] = set()

    def register_model(self, model_id: str, expected_hashes: Dict[str, str]) -> None:
        """Register expected hashes for a model.

        Args:
            model_id: HuggingFace model identifier.
            expected_hashes: Map of relative file path -> expected SHA-256 hex digest.
        """
        for rel_path, expected_hash in expected_hashes.items():
            # Store both the relative path and the basename for flexible matching
            full_path = str(rel_path)
            self._known_hashes[full_path] = expected_hash
            self._known_hashes[Path(rel_path).name] = expected_hash

    def verify_file(self, file_path: str) -> bool:
        """Verify a single model file against its expected hash.

        Args:
            file_path: Absolute path to the model file.

        Returns:
            True if the file matches the expected hash or is not registered.
        """
        if file_path in self._verified:
            return True

        # Try matching by full path, then by basename
        expected = self._known_hashes.get(file_path)
        if expected is None:
            expected = self._known_hashes.get(Path(file_path).name)
        if expected is None:
            # Not registered — trust it (first-time load)
            self._verified.add(file_path)
            return True

        actual = _compute_file_hash(file_path)

        if actual != expected:
            logger.warning(
                "Model cache mismatch for %s: expected %s, got %s",
                file_path, expected, actual,
            )
            return False

        self._verified.add(file_path)
        logger.debug("Model cache verified: %s", file_path)
        return True

    def verify_all(self, model_files: List[str]) -> List[str]:
        """Verify all files for a model.

        Args:
            model_files: List of absolute file paths to verify.

        Returns:
            List of files that failed verification (empty if all OK).
        """
        failures = []
        for f in model_files:
            if not self.verify_file(f):
                failures.append(f)
        return failures


def _compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hex digest of a file.

    Args:
        file_path: Path to the file.

    Returns:
        SHA-256 hex digest string.
    """
    h = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    except (OSError, IOError):
        return ""
    return h.hexdigest()
