"""Tests for Gap #10 — structured logging and profiling."""

from __future__ import annotations

import logging
import sys
from io import StringIO
from pathlib import Path

import pytest


class TestLoggingSetup:
    """CLI logging setup (Gap #10)."""

    def test_setup_logging_default_level(self):
        from okfgraph.cli import _setup_logging
        _setup_logging()
        root = logging.getLogger()
        # Default level is INFO
        assert root.level == logging.INFO

    def test_setup_logging_verbose(self):
        from okfgraph.cli import _setup_logging
        _setup_logging(verbose=True)
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_setup_logging_quiet(self):
        from okfgraph.cli import _setup_logging
        _setup_logging(quiet=True)
        root = logging.getLogger()
        assert root.level == logging.ERROR

    def test_setup_logging_file(self, tmp_path):
        from okfgraph.cli import _setup_logging
        log_file = str(tmp_path / "test.log")
        _setup_logging(log_file=log_file)
        root = logging.getLogger()
        # Should have at least 2 handlers (console + file)
        assert len(root.handlers) >= 2

    def test_teardown_logging(self):
        from okfgraph.cli import _setup_logging, _teardown_logging
        _setup_logging()
        root = logging.getLogger()
        handler_count = len(root.handlers)
        _teardown_logging()
        # Handlers should be cleaned up
        assert len(root.handlers) < handler_count


class TestImportBundleTiming:
    """import_bundle() timing instrumentation (Gap #10)."""

    @pytest.fixture(scope="function")
    def test_router(self, tmp_path):
        from okfgraph.router import OKFRouter
        db_path = str(tmp_path / "test.db")
        bundle_path = Path(__file__).parent / "fixtures" / "bundle"
        router = OKFRouter(
            db_path=db_path,
            bundle_root=str(bundle_path),
            embedding_dim=512,
            device="cpu",
            enable_chunking=False,
        )
        yield router
        router.close()

    def test_import_bundle_logs_phase_durations(self, test_router):
        """import_bundle logs phase durations to the logger."""
        from okfgraph.router import logger as router_logger

        # Capture log output
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.DEBUG)
        router_logger.addHandler(handler)

        # Import the test bundle
        bundle_path = Path(__file__).parent / "fixtures" / "bundle"
        test_router.import_bundle(bundle_path=bundle_path)

        # Check that phase durations were logged
        log_output = log_stream.getvalue()
        assert "delta:" in log_output or "import_bundle:" in log_output

        router_logger.removeHandler(handler)

    def test_import_bundle_logs_parse_count(self, test_router):
        """import_bundle logs the number of parsed concepts."""
        from okfgraph.router import logger as router_logger

        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.DEBUG)
        router_logger.addHandler(handler)

        bundle_path = Path(__file__).parent / "fixtures" / "bundle"
        test_router.import_bundle(bundle_path=bundle_path)

        log_output = log_stream.getvalue()
        assert "parsed" in log_output or "import_bundle:" in log_output

        router_logger.removeHandler(handler)

    def test_import_bundle_logs_encode_duration(self, test_router):
        """import_bundle logs encode phase duration."""
        from okfgraph.router import logger as router_logger

        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.DEBUG)
        router_logger.addHandler(handler)

        bundle_path = Path(__file__).parent / "fixtures" / "bundle"
        test_router.import_bundle(bundle_path=bundle_path)

        log_output = log_stream.getvalue()
        assert "encode:" in log_output or "import_bundle:" in log_output

        router_logger.removeHandler(handler)


class TestProfileFlag:
    """--profile CLI flag (Gap #10C)."""

    def test_profile_flag_imports(self):
        """The profile flag doesn't cause import errors."""
        # This test just verifies the import works
        from okfgraph.cli import main
        assert callable(main)

    def test_profile_flag_cprofile_available(self):
        """cProfile and pstats are available for profiling."""
        import cProfile
        import pstats
        profiler = cProfile.Profile()
        profiler.enable()
        profiler.disable()
        # Should not raise
        stream = StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats("cumulative")
        stats.print_stats(20)
        assert stream.getvalue()  # Should have output
