"""Functional tests for okf_ingest_tool — stubs the sandbox and the router so
the tool's own behaviour (extraction, okf-asset:// handoff, delegation, stats)
is exercised without torch / ladybug / okfgraph installed."""

import base64
import importlib.util
import re
import sys
from pathlib import Path

_TOOL_PATH = Path(__file__).resolve().parents[1] / "examples" / "okf_ingest_tool.py"
_spec = importlib.util.spec_from_file_location("okf_ingest_tool_under_test", _TOOL_PATH)
tool = importlib.util.module_from_spec(_spec)
sys.modules["okf_ingest_tool_under_test"] = tool
_spec.loader.exec_module(tool)

PNG_1x1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


class StubSandbox:
    """Minimal FileToolSandbox stand-in (no path confinement)."""

    def __init__(self, root: Path, dry_run: bool = False):
        self.root_dir = root.resolve()
        self.dry_run = dry_run

    def check_read(self): pass
    def check_write(self): pass

    def resolve_path(self, p: str) -> Path:
        cand = Path(p)
        return cand.resolve() if cand.is_absolute() else (self.root_dir / cand).resolve()


class FakeRouter:
    """Records the mode it was handed and mimics the router's per-route stats."""

    last_instance = None

    def __init__(self, db_path: str, bundle_root: str):
        self.bundle_root = Path(bundle_root)
        self.mode = None
        self.concepts = {}
        FakeRouter.last_instance = self

    def import_bundle(self, bundle_path=None, mode="text"):
        self.mode = mode
        root = Path(bundle_path or self.bundle_root)
        ids = []
        for md in sorted(root.rglob("*.md")):
            cid = str(md.relative_to(root)).replace(".md", "")
            body = md.read_text(encoding="utf-8")
            refs = re.findall(r'!\[(.*?)\]\(okf-asset://([^)]+)\)', body)
            self.concepts[cid] = [(alt.strip(), aid) for alt, aid in refs]
            ids.append(cid)
        return ids

    def list_images(self, cid):
        out = []
        for alt, aid in self.concepts.get(cid, []):
            if self.mode == "text":
                route = "text"
            elif self.mode == "omni":
                route = "omni"
            else:  # optional: omni only when alt-text is absent
                route = "text" if alt else "omni"
            out.append({"id": aid, "file_name": f"{aid}.png",
                        "alt_text": alt, "embed_route": route})
        return out


class FakePdfDoc:
    """Stand-in for pdf_oxide.PdfDocument: writes images to image_out_dir."""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def to_markdown_all(self, preserve_layout, detect_headings,
                        include_images, embed_images, image_out_dir):
        out = Path(image_out_dir)
        (out / "im0.png").write_bytes(PNG_1x1)
        (out / "im1.png").write_bytes(PNG_1x1)
        return "# Title\n\n![a labelled chart](im0.png)\n\nText.\n\n![](im1.png)\n"


def _setup(tmp: Path):
    src = tmp / "collection"
    src.mkdir(parents=True)
    # A single binary document; its images are produced by the (stubbed) converter.
    (src / "doc1.pdf").write_bytes(b"%PDF-1.4 fake")
    return src


def _run(tmp, **kw):
    tool.HAS_OKFGRAPH = True
    tool.OKFRouter = FakeRouter
    tool.PdfDocument = FakePdfDoc
    FakeRouter.last_instance = None
    sb = StubSandbox(tmp, dry_run=kw.pop("dry_run", False))
    return tool._ingest_to_okf_impl(
        sb, "collection", "okf_bundle", "okfgraph.db", "document", ["t"],
        kw.pop("image_mode", None), kw.pop("clip_mode", "on_missing_description"),
    )


def test_mode_resolution():
    assert tool._resolve_mode("omni", "never") == "omni"
    assert tool._resolve_mode(None, "always") == "omni"
    assert tool._resolve_mode(None, "never") == "text"
    assert tool._resolve_mode(None, "on_missing_description") == "optional"
    assert tool._resolve_mode(None, None) == "optional"
    try:
        tool._resolve_mode("banana", None)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_invalid_mode_returns_error(tmp_path):
    _setup(tmp_path)
    resp = _run(tmp_path, image_mode="banana")
    assert resp.processed_count == 0 and resp.errors
    assert "Invalid image_mode" in resp.errors[0]


def test_extracts_to_asset_store_and_rewrites_links(tmp_path):
    _setup(tmp_path)
    resp = _run(tmp_path, image_mode="optional")

    # Two images pulled out of the document.
    assert resp.images_extracted == 2

    md = (tmp_path / "okf_bundle" / "doc1.md").read_text()
    ids = re.findall(r'okf-asset://(img_[0-9a-f]+)', md)
    assert len(ids) == 2                       # both links rewritten to okf-asset://
    assert "im0.png" not in md and "im1.png" not in md

    # Bytes were staged into _assets/<id>.<ext> so the router can resolve them.
    assets = tmp_path / "okf_bundle" / "_assets"
    for aid in ids:
        assert (assets / f"{aid}.png").is_file()

    # The router was handed the chosen mode and did the embedding/linking.
    assert FakeRouter.last_instance.mode == "optional"
    assert resp.image_mode == "optional"
    assert resp.ingested_count == 1
    assert resp.images_ingested == 2
    # optional mode: labelled image -> text, unlabelled -> omni
    assert resp.images_via_text == 1
    assert resp.images_via_omni == 1
    assert resp.clip_embeddings_generated == resp.images_via_omni


def test_omni_mode_routes_all_images_to_omni(tmp_path):
    _setup(tmp_path)
    resp = _run(tmp_path, image_mode="omni")
    assert FakeRouter.last_instance.mode == "omni"
    assert resp.images_via_omni == 2 and resp.images_via_text == 0


def test_deterministic_ids_across_runs(tmp_path):
    _setup(tmp_path)
    md1 = _run(tmp_path, image_mode="text")
    ids1 = sorted(re.findall(r'okf-asset://(img_[0-9a-f]+)',
                             (tmp_path / "okf_bundle" / "doc1.md").read_text()))
    # Re-run into a fresh bundle dir name by clearing the previous bundle.
    import shutil
    shutil.rmtree(tmp_path / "okf_bundle")
    _run(tmp_path, image_mode="text")
    ids2 = sorted(re.findall(r'okf-asset://(img_[0-9a-f]+)',
                             (tmp_path / "okf_bundle" / "doc1.md").read_text()))
    assert ids1 == ids2 and len(ids1) == 2


def run():
    import inspect
    import tempfile
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        if "tmp_path" in inspect.signature(fn).parameters:
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d))
        else:
            fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")


if __name__ == "__main__":
    run()
