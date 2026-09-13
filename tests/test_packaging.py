"""Packaging metadata tests — PyPI project description + external embed dep."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_published_readme_is_declared():
    """The PyPI project declares its README as the long description."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["readme"] == "README.md"
    assert (ROOT / "README.md").is_file()


def test_embedding_dep_is_external_pin():
    """embroider is a plain PyPI pin — no path dep, no in-tree crate.

    Since 0.2.12 the embedding engine lives in the separate `embroider`
    repo (github.com/opticsWolf/embroider, wheels on PyPI, floor-pinned
    <0.2 to keep the Jina contract and wheel matrix in lock-step with
    okfgraph 0.2.x).
    """
    raw = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "tool.uv.sources" not in raw, "embroider path dep must stay gone"
    deps = tomllib.loads(raw)["project"]["dependencies"]
    assert any(d.startswith("embroider>=") for d in deps), deps
    assert not (ROOT / "rust" / "okf-embed").exists(), "in-tree crate must stay deleted"
