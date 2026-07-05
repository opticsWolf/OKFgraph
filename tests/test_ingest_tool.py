"""Stubbed functional test for okf_ingest_tool — no heavy deps, no real DB.

Verifies the tool: (1) extracts an image into the bundle's _assets store,
(2) rewrites the markdown link to okf-asset://<id>, (3) writes the concept .md,
(4) delegates embedding/storage/linking to the router (no DB work of its own),
and (5) maps legacy clip_mode onto the unified image mode.
"""
import base64
import importlib.util
import sys
import types
from pathlib import Path

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

# --- Load the tool module by path (its heavy deps are optional/guarded) ------
TOOL_PATH = Path(__file__).resolve().parents[1] / "examples" / "okf_ingest_tool.py"
spec = importlib.util.spec_from_file_location("okf_ingest_tool_under_test", TOOL_PATH)
tool = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tool
spec.loader.exec_module(tool)


# --- Fakes -------------------------------------------------------------------
class FakeSandbox:
    def __init__(self, root: Path):
        self.root_dir = root
        self.dry_run = False

    def check_read(self): pass
    def check_write(self): pass

    def resolve_path(self, p: str) -> Path:
        cand = Path(p)
        return cand if cand.is_absolute() else (self.root_dir / cand).resolve()


class FakeRouter:
    """Records the mode it was imported with and reports image embed routes."""
    last = {}

    def __init__(self, db_path, bundle_root, **kw):
        self.bundle_root = Path(bundle_root)
        FakeRouter.last["init"] = {"db_path": db_path, "bundle_root": bundle_root, **kw}

    def import_bundle(self, bundle_path=None, mode="text", **kw):
        FakeRouter.last["mode"] = mode
        root = Path(bundle_path or self.bundle_root)
        self._ids = [p.stem for p in sorted(root.glob("*.md"))]
        # capture the rewritten bodies for assertions
        FakeRouter.last["bodies"] = {p.stem: p.read_text() for p in sorted(root.glob("*.md"))}
        FakeRouter.last["assets"] = sorted(q.name for q in (root / "_assets").glob("*")) if (root / "_assets").is_dir() else []
        return self._ids

    def list_images(self, cid):
        # Pretend each concept's images were embedded via the omni route.
        body = FakeRouter.last["bodies"].get(cid, "")
        n = body.count("okf-asset://")
        return [{"id": f"img_{i}", "embed_route": "omni", "file_name": "x.png"} for i in range(n)]


def _run(tmp: Path, *, clip_mode="on_missing_description", image_mode=None):
    src = tmp / "collection"
    src.mkdir(parents=True, exist_ok=True)
    # a .txt "document" (no oxide needed) that references a sibling image
    (src / "doc1.txt").write_text("# Title\n\nText.\n\n![a cat](cat.png)\n\nmore\n")
    (src / "cat.png").write_bytes(PNG)

    sandbox = FakeSandbox(tmp)
    tool.OKFRouter = FakeRouter          # inject fake router
    tool.HAS_OKFGRAPH = True
    FakeRouter.last = {}

    return tool._ingest_to_okf_impl(
        sandbox, "collection", "okf_bundle", "okfgraph.db",
        "document", ["t1"], image_mode, clip_mode,
    )


def test_legacy_clip_mode_maps_to_optional():
    assert tool._resolve_mode(None, "on_missing_description") == "optional"
    assert tool._resolve_mode(None, "always") == "omni"
    assert tool._resolve_mode(None, "never") == "text"
    assert tool._resolve_mode("omni", "never") == "omni"   # explicit wins


def test_staging_and_delegation(tmp_path):
    resp = _run(tmp_path)
    bundle = tmp_path / "okf_bundle"

    # 1. concept md written
    md = (bundle / "doc1.md").read_text()
    # 2. link rewritten to okf-asset://, original local path gone
    assert "okf-asset://img_" in md
    assert "cat.png" not in md.split("okf-asset")[0].rsplit("(", 1)[-1]
    # 3. bytes staged into _assets/<id>.<ext>
    staged = list((bundle / "_assets").glob("img_*.png"))
    assert len(staged) == 1
    assert staged[0].read_bytes() == PNG
    # 4. router was asked to do the embedding, with the mapped mode
    assert FakeRouter.last["mode"] == "optional"
    # 5. tool did NOT create its own ImageAsset/embeddings — it reported the
    #    router's counts back
    assert resp.images_extracted == 1
    assert resp.ingested_count == 1
    assert resp.image_mode == "optional"
    assert resp.images_via_omni == 1


def test_explicit_omni_mode(tmp_path):
    _run(tmp_path, image_mode="omni")
    assert FakeRouter.last["mode"] == "omni"


def run():
    import inspect
    import tempfile
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        if "tmp_path" in inspect.signature(fn).parameters:
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d))
        else:
            fn()
        passed += 1
        print(f"  PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    run()
