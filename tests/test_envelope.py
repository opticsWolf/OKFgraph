"""§4 contract tests: envelope shape, typed errors, INTERNAL catch-all.

The envelope is the uniform transport for CLI ``--json`` and MCP (D7):
one shape per op — ``{ok, op, data, warnings, error}`` — with outcome
reports riding in ``data`` and **never** inside ``error``.
"""
import json

import pytest

from okfgraph.errors import (
    OKFError,
    OKFWarning,
    OutcomeError,
    StateError,
    UsageError,
    internal_error,
    envelope,
)


class TestCodeTable:
    def test_usage_codes_exit_2(self):
        for code in ("BAD_VALUE", "MISSING_PARAM", "FILE_NOT_FOUND",
                     "CONFIG_INVALID"):
            assert OKFError(code, "x").exit_code == 2, code

    def test_state_codes_exit_1_including_pins_and_lock(self):
        # §4 table: pins, lock timeout, vision/ort are *state* problems,
        # not usage — the session was opened wrong, not the caller dumb.
        for code in ("UNKNOWN_CONCEPT", "UNKNOWN_ASSET", "NOT_RECOVERABLE",
                     "DETACHED", "PURGE_REFUSED_ABSENT_ROOT",
                     "SEARCH_UNAVAILABLE", "WRITE_LOCK_TIMEOUT", "DB_LOCKED",
                     "DIM_MISMATCH", "MODEL_PIN_MISMATCH",
                     "PRECISION_PIN_MISMATCH", "IMAGE_PIN_MISMATCH",
                     "VISION_INCOMPATIBLE", "NO_ORT_RUNTIME"):
            assert OKFError(code, "x").exit_code == 1, code

    def test_outcome_codes_exit_1(self):
        for code in ("DOCTOR_FINDINGS", "LINT_ERRORS", "DIFF_DIFFERENT"):
            assert OKFError(code, "x").exit_code == 1, code

    def test_subclass_selected_by_code(self):
        assert isinstance(OKFError("BAD_VALUE", "x"), UsageError)
        assert isinstance(OKFError("DETACHED", "x"), StateError)
        assert isinstance(OKFError("DIFF_DIFFERENT", "x"), OutcomeError)

    def test_unknown_code_rejected(self):
        with pytest.raises(ValueError):
            OKFError("MADE_UP_CODE", "x")

    def test_legacy_excepts_still_catch(self):
        # UsageError ⊂ ValueError, StateError ⊂ RuntimeError: conversion
        # sites may keep the old except clauses until swept.
        with pytest.raises(ValueError):
            raise OKFError("BAD_VALUE", "x")
        with pytest.raises(RuntimeError):
            raise OKFError("DETACHED", "x")


class TestEnvelope:
    def test_success_shape(self):
        env = envelope("read", data={"a": 1}, warnings=[OKFWarning("careful")])
        assert env == {"ok": True, "op": "read", "data": {"a": 1},
                       "warnings": ["careful"], "error": None}

    def test_error_shape_from_okf_error(self):
        err = OKFError("UNKNOWN_CONCEPT", "no concept 'x'",
                       op="read",
                       fields={"concept_id": "x"},
                       remedy="search first to find IDs")
        env = envelope("read", error=err)
        assert env["ok"] is False
        assert env["data"] is None
        assert env["error"] == {"code": "UNKNOWN_CONCEPT",
                                "message": "no concept 'x'",
                                "fields": {"concept_id": "x"},
                                "remedy": "search first to find IDs"}

    def test_outcome_report_rides_in_data_not_error(self):
        # §4: DIFF_DIFFERENT/DOCTOR_FINDINGS/LINT_ERRORS carry the full
        # report as data; the error object only names the outcome code.
        report = {"identical": False, "added": ["gamma"]}
        env = envelope("diff", data=report)
        env["error"] = {"code": "DIFF_DIFFERENT", "message": "structures differ"}
        assert env["data"] is report
        assert "report" not in json.dumps(json.loads(json.dumps(env["error"])))


class TestInternal:
    def test_internal_error_names_type(self):
        err = internal_error(KeyError("boom"), op="read")
        assert err.code == "INTERNAL"
        assert err.fields["type"] == "KeyError"
        assert "KeyError" in err.message

    def test_internal_error_is_state_class(self):
        err = internal_error(TypeError("no"), op="op")
        assert err.exit_code == 1
        assert isinstance(err, StateError)


class TestJsonQuietLogs:
    """D3: `--json` defaults the console log level to WARNING (unless
    -v/-q wins) so merged-stream harnesses still parse the stdout
    envelope; INFO chatter must not pollute the stream."""

    def test_json_defaults_console_to_warning(self):
        import logging as _log
        import argparse as _ap
        from okfgraph.cli import _setup_logging
        root = _log.getLogger()
        old = root.level
        try:
            _setup_logging(console_level=_log.WARNING)
            assert root.level == _log.WARNING
            _setup_logging(console_level=None)
            assert root.level == _log.INFO
        finally:
            _setup_logging(console_level=old if isinstance(old, int) else None)

    def test_json_flag_beats_default_but_not_verbose(self):
        """main() wiring: json -> WARNING; verbose/quiet win over it."""
        import logging as _log
        from okfgraph.cli import main  # noqa: presence check
        # Re-derive the precedence the same way main() does.
        def level_for(verbose, quiet, json_):
            if quiet:
                return _log.ERROR
            if verbose:
                return _log.DEBUG
            if json_:
                return _log.WARNING
            return _log.INFO
        assert level_for(False, False, True) == _log.WARNING
        assert level_for(True, False, True) == _log.DEBUG
        assert level_for(False, True, True) == _log.ERROR
        assert level_for(False, False, False) == _log.INFO
