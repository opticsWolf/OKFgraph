"""Build/install the native PDF add-on wheel.

The plugin is a plain cdylib, not a PyO3 module, so maturin cannot package it.
This module is both a PEP 517 backend (``pip wheel ./addons/pdf``) and a CLI
(``python addons/pdf/build_wheel.py --profile dev``). The wheel contains the
``okfgraph_pdf`` locator plus the platform's native library.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import tomllib
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PYPROJECT = HERE / "pyproject.toml"
LOCATOR = ROOT / "python" / "okfgraph_pdf" / "__init__.py"
WHEELS = ROOT / "target" / "wheels"
LIBRARY_NAMES = (
    "okfgraph_pdf.dll",
    "libokfgraph_pdf.so",
    "libokfgraph_pdf.dylib",
)


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _project() -> dict:
    return _pyproject()["project"]


def _version() -> str:
    return str(_project()["version"])


def _normalized(name: str) -> str:
    return name.replace("-", "_").replace(".", "_")


def _platform_tag() -> str:
    platform = sysconfig.get_platform()
    if platform.startswith("win-amd64"):
        return "win_amd64"
    if platform.startswith("linux-x86_64"):
        # Local wheel; CI should auditwheel-repair before publishing.
        return "linux_x86_64"
    if platform.startswith("macosx-"):
        _, version, arch = platform.split("-", 2)
        major, minor = version.split(".", 1)
        return f"macosx_{major}_{minor}_{arch}"
    raise SystemExit(f"unsupported wheel platform: {platform}")


def _wheel_name() -> str:
    project = _project()
    return (
        f"{_normalized(str(project['name']))}-{_version()}"
        f"-py3-none-{_platform_tag()}.whl"
    )


def _metadata() -> str:
    project = _project()
    license_expression = project.get("license")
    lines = [
        "Metadata-Version: 2.4",
        f"Name: {project['name']}",
        f"Version: {_version()}",
        f"Summary: {project['description']}",
        f"Requires-Python: {project['requires-python']}",
    ]
    if isinstance(license_expression, str):
        lines.append(f"License-Expression: {license_expression}")
    lines.append("")
    return "\n".join(lines)


def _wheel_metadata() -> str:
    return "\n".join([
        "Wheel-Version: 1.0",
        "Generator: okfgraph-pdf build_wheel.py",
        "Root-Is-Purelib: false",
        f"Tag: py3-none-{_platform_tag()}",
        "",
    ])


def _cargo_library(profile: str) -> Path:
    command = ["cargo", "build", "-p", "okfgraph-pdf"]
    if profile != "dev":
        command += ["--profile", profile]
    subprocess.run(command, cwd=ROOT, check=True)
    target = ROOT / "target" / ("debug" if profile == "dev" else profile)
    for name in LIBRARY_NAMES:
        candidate = target / name
        if candidate.is_file():
            return candidate
    raise SystemExit(f"cargo did not produce a PDF plugin under {target}")


def _record(rows: list[tuple[str, bytes]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    for name, data in rows:
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
        writer.writerow([name, f"sha256={digest.decode('ascii')}", len(data)])
    dist_info = f"{_normalized(str(_project()['name']))}-{_version()}.dist-info"
    writer.writerow([f"{dist_info}/RECORD", "", ""])
    return output.getvalue()


def build_wheel(
    wheel_directory: str | os.PathLike,
    config_settings: dict | None = None,
    metadata_directory: str | os.PathLike | None = None,
) -> str:
    profile = "dev"
    if config_settings and config_settings.get("profile"):
        profile = str(config_settings["profile"])
    if profile == "dev" and os.environ.get("OKFGRAPH_PDF_RELEASE") == "1":
        profile = "release"
    library = _cargo_library(profile)
    wheel_directory = Path(wheel_directory)
    wheel_directory.mkdir(parents=True, exist_ok=True)
    wheel_path = wheel_directory / _wheel_name()
    dist_info = f"{_normalized(str(_project()['name']))}-{_version()}.dist-info"
    module = str(
        _pyproject()
        .get("tool", {})
        .get("okfgraph-pdf", {})
        .get("module", "okfgraph_pdf")
    )

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        package = stage / module
        package.mkdir(parents=True)
        shutil.copyfile(LOCATOR, package / "__init__.py")
        shutil.copyfile(library, package / library.name)
        info = stage / dist_info
        info.mkdir()
        entries: list[tuple[str, bytes]] = []
        for name, text in (
            ("METADATA", _metadata()),
            ("WHEEL", _wheel_metadata()),
        ):
            data = text.encode("utf-8")
            (info / name).write_bytes(data)
            entries.append((f"{dist_info}/{name}", data))
        for path in sorted(package.rglob("*")):
            if path.is_file():
                data = path.read_bytes()
                entries.append((str(path.relative_to(stage)).replace("\\", "/"), data))
        record = _record(entries).encode("utf-8")
        (info / "RECORD").write_bytes(record)
        entries.append((f"{dist_info}/RECORD", record))

        with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as wheel:
            for name, data in sorted(entries):
                wheel.writestr(name, data)
    print(f"Built wheel for {_platform_tag()}: {wheel_path}")
    return wheel_path.name


def prepare_metadata_for_build_wheel(
    metadata_directory: str | os.PathLike,
    config_settings: dict | None = None,
) -> str:
    dist_info = f"{_normalized(str(_project()['name']))}-{_version()}.dist-info"
    info = Path(metadata_directory) / dist_info
    info.mkdir(parents=True, exist_ok=True)
    (info / "METADATA").write_text(_metadata(), encoding="utf-8")
    (info / "WHEEL").write_text(_wheel_metadata(), encoding="utf-8")
    return dist_info


def get_requires_for_build_wheel(config_settings: dict | None = None) -> list[str]:
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(WHEELS))
    ap.add_argument("--profile", default="dev")
    args = ap.parse_args()
    build_wheel(args.out, {"profile": args.profile})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
