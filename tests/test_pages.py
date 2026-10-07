"""Small source-level guards for the static landing page (no browser dependency)."""
import json
import re
import shlex
import tomllib
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from okfgraph.cli import build_parser

PAGES = Path(__file__).resolve().parents[1] / "pages"


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.links = []
        self.assets = []
        self.blocks = []
        self.code_attrs = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "a":
            self.links.append(attrs.get("href", ""))
        if tag == "link":
            self.assets.append(attrs["href"])
        if tag == "pre":
            self.current = ""
            self.code_attrs.append(attrs)

    def handle_data(self, data):
        if self.current is not None:
            self.current += data

    def handle_endtag(self, tag):
        if tag == "pre":
            self.blocks.append(self.current)
            self.current = None


def page():
    result = Page()
    result.feed((PAGES / "index.html").read_text(encoding="utf-8"))
    return result


def test_pages_local_links_and_assets():
    parsed = page()
    assert len(parsed.ids) == len(set(parsed.ids))
    for link in parsed.links:
        if link.startswith("#"):
            assert link[1:] in parsed.ids
    for asset in parsed.assets:
        assert (PAGES / urlsplit(asset).path).is_file()


def test_pages_stylesheet_is_versioned():
    stylesheet = next(a for a in page().assets if urlsplit(a).path == "style.css")
    assert parse_qs(urlsplit(stylesheet).query).get("v")


def test_pages_agent_memory_is_discoverable():
    parsed = page()
    assert "agent-memory" in parsed.ids
    assert "#agent-memory" in parsed.links
    html = (PAGES / "index.html").read_text(encoding="utf-8")
    assert "vault for lasting agent memory" in html


def test_pages_cli_examples_parse():
    count = 0
    for block in page().blocks:
        logical = re.sub(r"\\\r?\n\s*", " ", block)
        for line in logical.splitlines():
            if line.startswith("okf "):
                build_parser().parse_args(shlex.split(line)[1:])
                count += 1
    assert count >= 20


def test_pages_config_points_to_example_corpus():
    config = next(b for b in page().blocks if b.startswith("bundle_root ="))
    parsed = tomllib.loads(config)
    assert parsed["bundle_root"] == "kb"
    assert parsed["database"]["db_path"] == "kb.db"
    assert parsed["embedding"]["embedding_dim"] == 512
    assert parsed["embedding"]["device"] == "cpu"


def test_pages_mcp_config_is_valid_json():
    config = next(b for b in page().blocks if b.startswith("{"))
    server = json.loads(config)["mcpServers"]["okfgraph"]
    assert server["command"] == "okf-mcp"
    assert server["args"][::2] == ["--db-path", "--bundle-root"]


def test_pages_scrollable_code_is_keyboard_accessible():
    for attrs in page().code_attrs:
        assert attrs.get("tabindex") == "0"
        assert attrs.get("aria-label")
    css = (PAGES / "style.css").read_text(encoding="utf-8")
    assert ":focus-visible" in css
    assert "prefers-reduced-motion" in css


def _uv_lock_versions():
    """Every ``[[package]]`` name -> version pinned in uv.lock."""
    text = (Path(__file__).resolve().parents[1] / "uv.lock").read_text(
        encoding="utf-8")
    return dict(re.findall(
        r'\[\[package\]\]\nname = "([^"]+)"\nversion = "([^"]+)"',
        text))


def test_pages_published_stack_matches_pins():
    """The 'Published stack' table must track the release + locked pins.

    okfgraph follows pyproject (bump it in the release commit, tag before
    merging to main so the table never claims an unpublished version);
    every dependency follows uv.lock. CPU/GPU runtime parity is asserted
    too — the plan-onnx-only single-binary pin.
    """
    root = Path(__file__).resolve().parents[1]
    html = (PAGES / "index.html").read_text(encoding="utf-8")
    release = tomllib.loads(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["version"]
    locked = _uv_lock_versions()
    assert locked["onnxruntime"] == locked["onnxruntime-gpu"]
    shown = dict(re.findall(
        r'pypi\.org/project/([^/"]+)/">[^<]*</a></td><td>([^<]+)</td>',
        html))
    assert shown["okfgraph"] == release
    assert f'<span class="release">{release}</span>' in html  # hero badge
    for pkg in ("bobine", "embroider", "ladybug", "onnxruntime"):
        assert shown[pkg] == locked[pkg], (pkg, shown[pkg], locked[pkg])
