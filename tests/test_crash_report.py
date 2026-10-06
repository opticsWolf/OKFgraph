"""Crash reports: unexpected failures leave their full traceback on disk.

Field report: a batch of `okf ingest` runs kept `2>&1 | tail -n 3`, which
showed one frame of an import-time NameError and lost the cause. Now both
the INTERNAL catch-all and the start-up import guard write a crash file
and name it on the final `[ERROR]` line.
"""
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture()
def crash_dir(tmp_path, monkeypatch):
    d = tmp_path / "crash"
    monkeypatch.setenv("OKF_CRASH_DIR", str(d))
    return d


def _boom(args):
    def inner():
        raise NameError("name 'functools' is not defined")
    inner()


class TestCatchAll:
    def test_internal_writes_full_traceback(self, crash_dir, monkeypatch, capsys):
        from okfgraph import cli
        monkeypatch.setitem(cli._COMMANDS, "ingest", _boom)
        args = cli.build_parser().parse_args(
            ["ingest", "--kind", "thoughts", "--thoughts", "x", "--topic", "t"])
        rc = cli._main_catchall(args)
        err = capsys.readouterr().err
        assert rc == 1
        files = list(crash_dir.glob("okf-*.log"))
        assert len(files) == 1
        last = err.strip().splitlines()[-1]
        assert last.startswith("[ERROR] INTERNAL") and str(files[0]) in last
        text = files[0].read_text(encoding="utf-8")
        assert "command:  ingest" in text
        assert "in inner" in text and "in _boom" in text  # every frame kept
        assert "NameError: name 'functools' is not defined" in text

    def test_json_envelope_names_the_file(self, crash_dir, monkeypatch, capsys):
        import json
        from okfgraph import cli
        monkeypatch.setitem(cli._COMMANDS, "ingest", _boom)
        args = cli.build_parser().parse_args(
            ["ingest", "--kind", "thoughts", "--thoughts", "x", "--topic", "t",
             "--json"])
        cli._main_catchall(args)
        env = json.loads(capsys.readouterr().out)
        path = env["error"]["fields"]["crash_report"]
        assert Path(path).is_file()

    def test_unwritable_dir_degrades(self, tmp_path, monkeypatch, capsys):
        from okfgraph import cli
        blocker = tmp_path / "file"
        blocker.write_text("x")
        monkeypatch.setenv("OKF_CRASH_DIR", str(blocker / "sub"))
        monkeypatch.setitem(cli._COMMANDS, "ingest", _boom)
        args = cli.build_parser().parse_args(
            ["ingest", "--kind", "thoughts", "--thoughts", "x", "--topic", "t"])
        assert cli._main_catchall(args) == 1
        assert "traceback above" in capsys.readouterr().err


class TestStartupGuard:
    def test_import_failure_reported(self, crash_dir, monkeypatch, capsys):
        from okfgraph import _entry
        monkeypatch.setitem(sys.modules, "okfgraph.cli", None)  # import fails
        with pytest.raises(SystemExit) as exc:
            _entry.okf()
        assert exc.value.code == 1
        last = capsys.readouterr().err.strip().splitlines()[-1]
        assert "okf failed to start" in last
        files = list(crash_dir.glob("okf-*.log"))
        assert len(files) == 1 and str(files[0]) in last
        assert "(start-up)" in files[0].read_text(encoding="utf-8")

    def test_crash_module_is_light(self):
        """The guard only works if importing it leaves the router unloaded."""
        code = ("import sys, okfgraph._entry, okfgraph.crash; "
                "print('okfgraph.router' in sys.modules)")
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True, check=True)
        assert out.stdout.strip() == "False"

    def test_lazy_package_exports(self):
        import okfgraph
        from okfgraph.router import OKFRouter
        assert okfgraph.OKFRouter is OKFRouter
        assert callable(okfgraph.cli_main)
        with pytest.raises(AttributeError):
            okfgraph.nope


def test_prune_keeps_newest(crash_dir):
    from okfgraph import crash
    for _ in range(crash.KEEP + 5):
        try:
            raise RuntimeError("x")
        except RuntimeError as e:
            crash.write_crash_report(e, command="t")
    assert len(list(crash_dir.glob("okf-*.log"))) == crash.KEEP
