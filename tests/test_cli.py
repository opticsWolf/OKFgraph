"""CLI tests — verify all commands parse correctly and invoke without errors."""

import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


class TestCLIHelp:
    """Test CLI help output (no DB needed)."""

    def test_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "okfgraph.cli", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "OKF Knowledge Graph CLI" in result.stdout

    def test_init_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "okfgraph.cli", "init", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--db" in result.stdout

    def test_import_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "okfgraph.cli", "import", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "Files to import" in result.stdout

    def test_search_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "okfgraph.cli", "search", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "Search query" in result.stdout

    def test_traverse_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "okfgraph.cli", "traverse", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "start_id" in result.stdout

    def test_list_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "okfgraph.cli", "list", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "directory" in result.stdout

    def test_get_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "okfgraph.cli", "get", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "concept_id" in result.stdout

    def test_export_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "okfgraph.cli", "export", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--output" in result.stdout

    def test_shell_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "okfgraph.cli", "shell", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--db" in result.stdout

    def test_model_info_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "okfgraph.cli", "model-info", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--cache-dir" in result.stdout
        assert "--model-id" in result.stdout


class TestCLIFullWorkflow:
    """Test full CLI workflow with real DB and model."""

    @classmethod
    def setup_class(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.db_path = f"{cls.tmpdir}/test.db"
        cls.bundle = f"{cls.tmpdir}/bundle"
        Path(cls.bundle).mkdir()

        # Create sample OKF file
        fp = Path(cls.bundle) / "hello.md"
        fm = {"type": "note", "title": "Hello World", "description": "A greeting", "tags": ["test"]}
        yaml_str = yaml.dump(fm, default_flow_style=False, sort_keys=False)
        fp.write_text(f"---\n{yaml_str}---\n\n# Hello World\n\nThis is a test concept.\n")

        # Init
        cls._run(["init", "--db", cls.db_path, "--bundle", cls.bundle])
        assert Path(cls.db_path).exists()

        # Import
        cls._run(["import", "--db", cls.db_path, "--bundle", cls.bundle, str(fp)])

        # Export single
        cls.export_dir = f"{cls.tmpdir}/exported"
        cls._run(["export", "--db", cls.db_path, "--bundle", cls.bundle, "--output", cls.export_dir, "--concept-id", "hello"])

        # Export bundle
        cls.export_bundle_dir = f"{cls.tmpdir}/exported_bundle"
        cls._run(["export", "--db", cls.db_path, "--bundle", cls.bundle, "--all", "--output", cls.export_bundle_dir])

    @classmethod
    def _run(cls, args):
        result = subprocess.run(
            [sys.executable, "-m", "okfgraph.cli"] + args,
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
        return result

    def test_init_creates_db(self):
        assert Path(self.db_path).exists()

    def test_search_finds_result(self):
        result = self._run(["search", "--db", self.db_path, "--bundle", self.bundle, "greeting"])
        assert "Hello World" in result.stdout

    def test_list_shows_root(self):
        result = self._run(["list", "--db", self.db_path, "--bundle", self.bundle])
        assert "Hello World" in result.stdout

    def test_get_returns_json(self):
        result = self._run(["get", "--db", self.db_path, "--bundle", self.bundle, "hello"])
        assert '"title"' in result.stdout
        assert "Hello World" in result.stdout

    def test_export_single_creates_file(self):
        exported = Path(self.export_dir) / "hello.md"
        assert exported.exists()
        content = exported.read_text()
        assert "Hello World" in content

    def test_export_bundle_creates_files(self):
        files = list(Path(self.export_bundle_dir).rglob("*.md"))
        assert len(files) >= 1

    def test_traverse_runs(self):
        result = self._run(["traverse", "--db", self.db_path, "--bundle", self.bundle, "hello"])
        assert result.returncode == 0

    def test_broken_links_help(self):
        result = self._run(["broken-links", "--help"])
        assert result.returncode == 0
        assert "--db" in result.stdout

    def test_repair_links_help(self):
        result = self._run(["repair-links", "--help"])
        assert result.returncode == 0
        assert "--db" in result.stdout
