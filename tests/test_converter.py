"""Verify the GUI converter's image staging produces okfgraph-ready output.

PySide6 is stubbed so the module imports without a display/Qt. We exercise the
pure staging helper, then feed its output to okfgraph.images so the writer
(converter) and reader (router) are proven to agree on the okf-asset://
+ _assets/<id>.<ext> contract.
"""
import base64
import importlib.util
import sys
import types
from pathlib import Path

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


class _Dummy:
    """Permissive stand-in usable as a base class, callable, and attr source."""
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return self

    def __getattr__(self, name):
        return _Dummy()


def _install_fake_pyside6():
    names = {
        "PySide6.QtWidgets": [
            "QApplication", "QMainWindow", "QVBoxLayout", "QHBoxLayout",
            "QLabel", "QTextBrowser", "QWidget", "QFileDialog", "QMessageBox",
            "QPushButton", "QProgressBar", "QFrame", "QCheckBox",
        ],
        "PySide6.QtCore": ["Qt", "QThread", "Signal"],
        "PySide6.QtGui": ["QDragEnterEvent", "QDropEvent"],
    }
    root = types.ModuleType("PySide6")
    sys.modules["PySide6"] = root
    for mod_name, attrs in names.items():
        m = types.ModuleType(mod_name)
        for a in attrs:
            setattr(m, a, _Dummy)
        sys.modules[mod_name] = m
        setattr(root, mod_name.split(".")[-1], m)


_install_fake_pyside6()

# Load the converter (Qt now satisfied; pdf_oxide/office_oxide stay None).
CONV_PATH = Path(__file__).resolve().parents[1] / "examples" / "oxide_to_markdown_converter.py"
spec = importlib.util.spec_from_file_location("oxide_converter_under_test", CONV_PATH)
conv = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = conv
spec.loader.exec_module(conv)

# Load okfgraph.images standalone (the reader side).
IMG_PATH = Path(__file__).resolve().parents[1] / "okfgraph" / "images.py"
ispec = importlib.util.spec_from_file_location("okfgraph_images_reader", IMG_PATH)
images = importlib.util.module_from_spec(ispec)
sys.modules[ispec.name] = images
ispec.loader.exec_module(images)


def test_asset_id_matches_ingest_tool_scheme():
    # img_<sha256(stem|occurrence|bytes)[:16]>
    aid = conv._asset_id("doc1", 1, PNG)
    assert aid.startswith("img_") and len(aid) == len("img_") + 16
    assert conv._asset_id("doc1", 1, PNG) == aid           # deterministic
    assert conv._asset_id("doc2", 1, PNG) != aid           # concept-scoped


def test_stage_and_round_trip(tmp_path):
    # A converted document dir with the extractor's temp images.
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    source = src_dir / "report.pdf"
    source.write_bytes(b"%PDF-1.4 fake")
    img_tmp = tmp_path / "imgtmp"
    img_tmp.mkdir()
    (img_tmp / "fig1.png").write_bytes(PNG)

    out_dir = tmp_path / "bundle"
    out_dir.mkdir()

    md = "# Report\n\n![Figure 1](fig1.png)\n\n![remote](https://x/y.png)\n"
    new_md, count = conv._stage_images_as_okf_assets(md, img_tmp, source, out_dir, "report")

    # 1. local image rewritten to okf-asset://; remote left alone
    assert count == 1
    assert "okf-asset://img_" in new_md
    assert "https://x/y.png" in new_md
    assert "fig1.png" not in new_md

    # 2. bytes staged into _assets/<id>.<ext>
    staged = list((out_dir / conv.ASSET_STORE_DIRNAME).glob("img_*.png"))
    assert len(staged) == 1 and staged[0].read_bytes() == PNG

    # 3. ROUND TRIP: the router's reader resolves the converter's okf-asset link
    imgs = images.build_extracted_images("report", new_md, search_dirs=[out_dir])
    okf = [i for i in imgs if i.src.startswith("okf-asset://")]
    assert len(okf) == 1
    resolved = okf[0]
    assert resolved.has_data and resolved.data == PNG     # reader found the bytes
    assert resolved.mime_type == "image/png"
    # id embedded in the link is exactly what the reader keys on
    assert resolved.asset_id == new_md.split("okf-asset://", 1)[1].split(")", 1)[0]


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
