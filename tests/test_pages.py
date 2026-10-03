"""Small source-level guards for the static landing page (no browser dependency)."""
import json
import re
import shlex
import tomllib
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from okfgraph.cli import build_parser
from okfgraph.config import OKFConfig

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
    config = next(b for b in page().blocks if b.startswith("bundle ="))
    parsed = OKFConfig._parse_toml(tomllib.loads(config))
    assert parsed.bundle == "kb"
    assert parsed.database.path == "kb.db"
    assert parsed.database.dim == 512
    assert parsed.embedding.device == "cpu"


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
