"""Tests for okf_search_browser — pure helpers + DbBackend with a fake router.

PySide6 is stubbed so the module imports without Qt; the database backend is a
plain class, so it is exercised directly with a fake router (no torch/ladybug).
"""
import base64
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


class _Dummy:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return self

    def __getattr__(self, name):
        return _Dummy()


class _QTextDocument:
    class ResourceType:
        ImageResource = 3


def _install_fake_qt():
    root = types.ModuleType("PySide6")
    sys.modules["PySide6"] = root
    widgets = [
        "QApplication", "QMainWindow", "QWidget", "QVBoxLayout", "QHBoxLayout",
        "QGridLayout", "QLabel", "QLineEdit", "QPushButton", "QComboBox",
        "QSpinBox", "QCheckBox", "QListWidget", "QListWidgetItem", "QTextBrowser",
        "QSplitter", "QFileDialog", "QMessageBox", "QFrame",
    ]
    core = ["Qt", "QThread", "Signal", "QByteArray"]
    spec = {"PySide6.QtWidgets": widgets, "PySide6.QtCore": core}
    for mod_name, attrs in spec.items():
        m = types.ModuleType(mod_name)
        for a in attrs:
            setattr(m, a, _Dummy)
        sys.modules[mod_name] = m
        setattr(root, mod_name.split(".")[-1], m)
    gui = types.ModuleType("PySide6.QtGui")
    gui.QImage = _Dummy
    gui.QTextDocument = _QTextDocument
    sys.modules["PySide6.QtGui"] = gui
    setattr(root, "QtGui", gui)


_install_fake_qt()

MOD_PATH = Path(__file__).resolve().parents[1] / "examples" / "okf_search_browser.py"
spec = importlib.util.spec_from_file_location("okf_search_browser_under_test", MOD_PATH)
ui = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ui
spec.loader.exec_module(ui)


# ── Fake router ─────────────────────────────────────────────────────────
class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def rows_as_dict(self):
        return SimpleNamespace(get_all=lambda: self._rows)


class FakeRouter:
    def __init__(self):
        self.conn = SimpleNamespace(
            execute=lambda q, p=None: FakeResult([{"id": "concepts/intro"}])
        )

    def get_by_id(self, cid):
        if cid == "missing":
            return None
        return SimpleNamespace(
            id=cid, title="Intro", type="document", tags=["x"],
            description="A concept.",
            body="See ![fig](okf-asset://img_abc) and ![two](okf-asset://img_def).",
        )

    def get_image_data(self, aid):
        if aid in ("img_abc", "img_def"):
            return {"id": aid, "file_name": f"{aid}.png", "mime_type": "image/png",
                    "alt_text": "a fig", "embed_route": "omni", "data": PNG}
        return None

    def search_hybrid(self, **kw):
        FakeRouter.last_hybrid = kw
        return [{"id": "c1", "title": "T", "type": "document",
                 "description": "d", "tags": [], "relevance_score": 0.42}]

    def search_images_with_text(self, **kw):
        FakeRouter.last_images = kw
        return [{"id": "img_abc", "file_name": "img_abc.png", "alt_text": "a fig",
                 "embed_route": "omni", "distance": 0.1, "relevance_score": 0.9}]


def _backend():
    b = ui.DbBackend()
    b.router = FakeRouter()
    b._cfg = {"db": "x", "bundle": "y", "dim": 512, "device": "cpu"}
    return b


# ── Tests ───────────────────────────────────────────────────────────────
def test_extract_asset_ids_order_dedup():
    text = "a ![x](okf-asset://img_1) b ![y](okf-asset://img_2) c okf-asset://img_1"
    assert ui.extract_asset_ids(text) == ["img_1", "img_2"]
    assert ui.extract_asset_ids("") == []


def test_result_labels():
    cl = ui.concept_result_label({"relevance_score": 0.5, "title": "Hi", "type": "doc"})
    assert "Hi" in cl and "doc" in cl and "0.500" in cl
    il = ui.image_result_label({"relevance_score": 0.9, "alt_text": "cat", "embed_route": "omni"})
    assert "cat" in il and "omni" in il


def test_build_concept_markdown():
    md = ui.build_concept_markdown({
        "id": "concepts/intro", "title": "Intro", "type": "document",
        "tags": ["a", "b"], "description": "Desc.", "body": "Body text here.",
    })
    assert md.startswith("# Intro")
    assert "**type:** document" in md
    assert "tags:** a, b" in md
    assert "> Desc." in md
    assert "Body text here." in md


def test_build_concept_markdown_missing():
    md = ui.build_concept_markdown({"id": "nope", "missing": True})
    assert "Not found" in md and "nope" in md


def test_build_image_markdown():
    md = ui.build_image_markdown(
        {"id": "img_abc", "file_name": "f.png", "mime_type": "image/png",
         "alt_text": "a cat", "embed_route": "omni"},
        including=["concepts/intro"],
    )
    assert "![a cat](okf-asset://img_abc)" in md
    assert "**embedded via:** omni" in md
    assert "`concepts/intro`" in md


def test_backend_preview_concept_resolves_images():
    b = _backend()
    payload = b.preview_concept("concepts/intro")
    assert payload["_kind"] == "concept"
    assert payload["title"] == "Intro"
    # both okf-asset ids in the body were resolved to bytes from the DB
    assert set(payload["images"]) == {"img_abc", "img_def"}
    assert payload["images"]["img_abc"] == PNG


def test_backend_preview_concept_missing():
    b = _backend()
    payload = b.preview_concept("missing")
    assert payload["missing"] is True and payload["images"] == {}


def test_backend_preview_image_with_including():
    b = _backend()
    payload = b.preview_image("img_abc")
    assert payload["_kind"] == "image"
    assert payload["data"] == PNG
    assert payload["meta"]["embed_route"] == "omni"
    assert payload["including"] == ["concepts/intro"]   # reverse INCLUDES_ASSET


def test_backend_search_tags_kind_and_maps_omni():
    b = _backend()
    cres = b.search_concepts("q", "document", ["t1"], "concepts", 5)
    assert cres[0]["_kind"] == "concept"
    assert FakeRouter.last_hybrid["concept_type"] == "document"
    assert FakeRouter.last_hybrid["tags"] == ["t1"]

    ires = b.search_images("q", use_omni=True, limit=3)
    assert ires[0]["_kind"] == "image"
    # use_omni=True  -> use_text_model=False
    assert FakeRouter.last_images["use_text_model"] is False


def run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    run()
