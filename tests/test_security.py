"""Tests for okfgraph.security — path traversal protection + cache verification."""

import hashlib
import ipaddress
import os
import tempfile
from pathlib import Path

import pytest

from okfgraph.security import (
    ModelCacheVerifier,
    is_path_safe,
    is_private_ip,
    validate_image_src,
)


class TestPrivateIPDetection:
    """Test that private/internal IPs are correctly identified."""

    def test_localhost_is_private(self):
        assert is_private_ip("localhost") is True
        assert is_private_ip("localhost.localdomain") is True

    def test_loopback_is_private(self):
        assert is_private_ip("127.0.0.1") is True
        assert is_private_ip("0.0.0.0") is True

    def test_private_ranges(self):
        assert is_private_ip("10.0.0.1") is True
        assert is_private_ip("192.168.1.1") is True
        assert is_private_ip("172.16.0.1") is True
        assert is_private_ip("172.31.255.255") is True

    def test_link_local_is_private(self):
        assert is_private_ip("169.254.169.254") is True  # AWS metadata

    def test_public_ip_is_not_private(self):
        assert is_private_ip("8.8.8.8") is False
        assert is_private_ip("1.1.1.1") is False

    def test_domain_name_is_not_private(self):
        assert is_private_ip("example.com") is False
        assert is_private_ip("cdn.example.com") is False


class TestPathTraversal:
    """Test that path traversal attacks are blocked."""

    def test_path_within_root_is_safe(self, tmp_path):
        inner = tmp_path / "subdir" / "file.txt"
        inner.parent.mkdir(parents=True, exist_ok=True)
        inner.touch()
        assert is_path_safe(inner, tmp_path) is True

    def test_path_outside_root_is_not_safe(self, tmp_path):
        outside = Path("/etc/passwd")
        assert is_path_safe(outside, tmp_path) is False

    def test_relative_path_escape_is_not_safe(self, tmp_path):
        malicious = tmp_path / "../../etc/passwd"
        assert is_path_safe(malicious, tmp_path) is False

    def test_symlink_escape_is_not_safe(self, tmp_path):
        # Create a file outside the bundle root
        outside = tmp_path / ".." / "outside_file.txt"
        outside.parent.mkdir(exist_ok=True)
        outside.touch()
        # Symlink inside the bundle pointing outside
        link = tmp_path / "link.txt"
        try:
            link.symlink_to(outside)
            assert is_path_safe(link, tmp_path) is False
        except OSError:
            pytest.skip("Symlinks not supported on this platform")


class TestImageSrcValidation:
    """Test that image source references are validated for security."""

    def test_okf_asset_is_safe(self, tmp_path):
        safe, reason = validate_image_src("okf-asset://abc123", tmp_path)
        assert safe is True
        assert "okf-asset" in reason

    def test_local_file_within_root_is_safe(self, tmp_path):
        safe, reason = validate_image_src("image.png", tmp_path)
        assert safe is True

    def test_path_traversal_is_blocked(self, tmp_path):
        safe, reason = validate_image_src("../etc/passwd", tmp_path)
        assert safe is False
        assert "outside bundle root" in reason

    def test_file_url_is_blocked(self, tmp_path):
        safe, reason = validate_image_src("file:///etc/passwd", tmp_path)
        assert safe is False
        assert "SSRF" in reason

    def test_remote_disabled_by_default(self, tmp_path):
        safe, reason = validate_image_src("https://example.com/img.png", tmp_path)
        assert safe is False
        assert "disabled" in reason

    def test_remote_allowed_with_allow_remote(self, tmp_path):
        safe, reason = validate_image_src(
            "https://example.com/img.png", tmp_path,
            allow_remote=True,
        )
        assert safe is True

    def test_remote_blocked_with_allowlist(self, tmp_path):
        safe, reason = validate_image_src(
            "https://evil.com/img.png", tmp_path,
            allow_remote=True,
            allowed_domains=["example.com"],
        )
        assert safe is False
        assert "not in allowlist" in reason

    def test_remote_allowed_with_matching_domain(self, tmp_path):
        safe, reason = validate_image_src(
            "https://cdn.example.com/img.png", tmp_path,
            allow_remote=True,
            allowed_domains=["*.example.com"],
        )
        assert safe is True

    def test_private_ip_blocked(self, tmp_path):
        safe, reason = validate_image_src(
            "http://169.254.169.254/latest/meta-data/", tmp_path,
            allow_remote=True,
        )
        assert safe is False
        assert "private IP" in reason


class TestModelCacheVerifier:
    """Test model cache integrity verification."""

    def test_verify_unregistered_file(self):
        verifier = ModelCacheVerifier()
        # Unregistered files are trusted (first-time load)
        assert verifier.verify_file("/nonexistent/file.onnx") is True

    def test_verify_registered_file_matches(self, tmp_path):
        # Create a test file
        test_file = tmp_path / "model.onnx"
        test_file.write_bytes(b"test model data")
        expected_hash = hashlib.sha256(b"test model data").hexdigest()

        verifier = ModelCacheVerifier()
        verifier.register_model("test-model", {"model.onnx": expected_hash})
        assert verifier.verify_file(str(test_file)) is True

    def test_verify_registered_file_mismatch(self, tmp_path):
        # Create a test file with wrong hash
        test_file = tmp_path / "model.onnx"
        test_file.write_bytes(b"corrupted data")
        wrong_hash = hashlib.sha256(b"expected data").hexdigest()

        verifier = ModelCacheVerifier()
        verifier.register_model("test-model", {"model.onnx": wrong_hash})
        assert verifier.verify_file(str(test_file)) is False

    def test_verify_all_returns_failures(self, tmp_path):
        # Create two files
        good_file = tmp_path / "good.onnx"
        bad_file = tmp_path / "bad.onnx"
        good_file.write_bytes(b"good data")
        bad_file.write_bytes(b"bad data")

        good_hash = hashlib.sha256(b"good data").hexdigest()
        bad_hash = hashlib.sha256(b"expected good data").hexdigest()

        verifier = ModelCacheVerifier()
        verifier.register_model("test-model", {
            "good.onnx": good_hash,
            "bad.onnx": bad_hash,
        })
        failures = verifier.verify_all([str(good_file), str(bad_file)])
        assert len(failures) == 1
        assert str(bad_file) in failures

    def test_caching_of_verified_files(self, tmp_path):
        test_file = tmp_path / "model.onnx"
        test_file.write_bytes(b"test data")
        expected_hash = hashlib.sha256(b"test data").hexdigest()

        verifier = ModelCacheVerifier()
        verifier.register_model("test-model", {"model.onnx": expected_hash})
        verifier.verify_file(str(test_file))
        assert str(test_file) in verifier._verified

    def test_nonexistent_file_returns_empty_hash(self):
        from okfgraph.security import _compute_file_hash
        assert _compute_file_hash("/nonexistent/file.onnx") == ""
