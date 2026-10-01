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
    <0.3 to keep the model registry and wheel matrix in lock-step with
    okfgraph 0.6.x).
    """
    raw = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "tool.uv.sources" not in raw, "embroider path dep must stay gone"
    deps = tomllib.loads(raw)["project"]["dependencies"]
    assert any(d.startswith("embroider>=") for d in deps), deps
    assert not (ROOT / "rust" / "okf-embed").exists(), "in-tree crate must stay deleted"


def test_no_torch_in_install_paths():
    """No install path may pull torch/transformers/optimum/sentence-transformers.

    Phase 0 guardrail (plan-onnx-only): the whole chain is ONNX-only.
    Test-only uses (parity tokenizer) live in tests/, never in packaging.
    FAILS on current dev via the `omni` extra — that is the point (Phase 1
    deletes the extra).
    """
    banned = ("torch", "transformers", "optimum", "sentence-transformers")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        declared.extend(extra)
    offenders = [d for d in declared if any(b in d.lower() for b in banned)]
    assert not offenders, f"torch-path packages in install paths: {offenders}"
