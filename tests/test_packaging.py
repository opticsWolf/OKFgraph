"""Packaging metadata tests — PyPI project descriptions."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_published_readmes_are_declared():
    """Both PyPI projects declare their README as the long description."""
    cases = [
        (ROOT / "pyproject.toml", "README.md"),
        (ROOT / "rust" / "okf-embed" / "pyproject.toml", "README.md"),
    ]
    for project_file, readme in cases:
        project = tomllib.loads(project_file.read_text(encoding="utf-8"))["project"]
        assert project["readme"] == readme
        assert (project_file.parent / readme).is_file()


def test_embed_python_and_rust_versions_match():
    """The okf-embed Python and Cargo versions stay in lock-step."""
    pyproject = tomllib.loads(
        (ROOT / "rust" / "okf-embed" / "pyproject.toml").read_text(encoding="utf-8")
    )
    cargo = tomllib.loads(
        (ROOT / "rust" / "okf-embed" / "Cargo.toml").read_text(encoding="utf-8")
    )
    assert pyproject["project"]["version"] == cargo["package"]["version"]
