"""Unit tests for okfgraph.images — pure logic, no heavy deps required."""

import base64
import importlib.util
import sys
from pathlib import Path

# Load okfgraph/images.py directly by path so this test does not pull in the
# heavy router/model stack via the package __init__.
_IMAGES_PATH = Path(__file__).resolve().parents[1] / "okfgraph" / "images.py"
_spec = importlib.util.spec_from_file_location("okfgraph_images_under_test", _IMAGES_PATH)
images = importlib.util.module_from_spec(_spec)
sys.modules["okfgraph_images_under_test"] = images  # required for @dataclass introspection
_spec.loader.exec_module(images)

EmbedRoute = images.EmbedRoute
IngestMode = images.IngestMode
ExtractedImage = images.ExtractedImage
asset_id_for = images.asset_id_for
build_extracted_images = images.build_extracted_images
extract_image_refs = images.extract_image_refs
fallback_caption = images.fallback_caption
filename_from_src = images.filename_from_src
plan_embedding = images.plan_embedding
sniff_mime = images.sniff_mime

PNG_1x1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def test_mode_coercion():
    assert IngestMode.coerce("text") is IngestMode.TEXT
    assert IngestMode.coerce("optional") is IngestMode.OPTIONAL
    assert IngestMode.coerce("full") is IngestMode.OMNI
    assert IngestMode.coerce("OMNI") is IngestMode.OMNI
    assert IngestMode.coerce(None) is IngestMode.TEXT
    assert IngestMode.coerce(None, default=IngestMode.OMNI) is IngestMode.OMNI
    assert IngestMode.coerce(IngestMode.OPTIONAL) is IngestMode.OPTIONAL
    try:
        IngestMode.coerce("banana")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown mode")


def test_extract_image_refs_order_and_alt():
    body = (
        "intro\n"
        "![first](a.png)\n"
        "text ![](b.jpg) inline\n"
        '![third pic](sub/c.gif "a title")\n'
        "[not an image](page.md)\n"
        "![remote](https://example.com/d.webp)\n"
        "![angle](<weird name.png>)\n"
    )
    refs = extract_image_refs(body)
    assert [src for _, src in refs] == [
        "a.png",
        "b.jpg",
        "sub/c.gif",
        "https://example.com/d.webp",
        "weird name.png",
    ]
    assert refs[0][0] == "first"
    assert refs[1][0] == ""          # empty alt
    assert refs[2][0] == "third pic"  # title stripped, alt kept


def test_filename_from_src():
    assert filename_from_src("sub/dir/photo.png") == "photo.png"
    assert filename_from_src("https://x.com/a/b/c.jpg?v=2#frag") == "c.jpg"
    assert filename_from_src("okf-asset://img_abc123") == "img_abc123"
    assert filename_from_src("with%20space.png") == "with space.png"
    assert filename_from_src("data:image/png;base64,AAAA").endswith(".png")


def test_sniff_mime():
    assert sniff_mime(PNG_1x1, "whatever.bin") == "image/png"
    assert sniff_mime(b"\xff\xd8\xff\xe0junk", "x") == "image/jpeg"
    assert sniff_mime(None, "pic.gif") == "image/gif"
    assert sniff_mime(b"not an image", "noext") == "application/octet-stream"


def test_fallback_caption():
    cap = fallback_caption("diagram.png", 3, "concepts/intro")
    assert "diagram.png" in cap
    assert "image 3" in cap
    assert "intro" in cap


def test_asset_id_deterministic_and_okf_passthrough():
    a = asset_id_for("c/x", 2, "p.png")
    b = asset_id_for("c/x", 2, "p.png")
    c = asset_id_for("c/x", 3, "p.png")
    assert a == b and a != c
    assert a.startswith("img_")
    assert asset_id_for("c/x", 1, "okf-asset://img_kept") == "img_kept"


def _img(alt=None, data=None, filename="f.png", index=1, concept_id="c"):
    return ExtractedImage(
        concept_id=concept_id, index=index, src="f.png",
        alt_text=alt, filename=filename, data=data,
    )


def test_plan_text_mode():
    # alt-text present -> embed alt-text
    route, cap = plan_embedding(_img(alt="a cat"), IngestMode.TEXT)
    assert route is EmbedRoute.TEXT and cap == "a cat"
    # no alt-text -> filename + image number fallback (never omni)
    route, cap = plan_embedding(_img(alt=None, data=PNG_1x1), IngestMode.TEXT)
    assert route is EmbedRoute.TEXT
    assert "f.png" in cap and "image 1" in cap


def test_plan_optional_mode():
    # alt-text -> text path
    route, cap = plan_embedding(_img(alt="chart", data=PNG_1x1), IngestMode.OPTIONAL)
    assert route is EmbedRoute.TEXT and cap == "chart"
    # no alt-text + bytes -> omni
    route, cap = plan_embedding(_img(alt=None, data=PNG_1x1), IngestMode.OPTIONAL)
    assert route is EmbedRoute.OMNI and cap is None
    # no alt-text + no bytes -> graceful text fallback
    route, cap = plan_embedding(_img(alt=None, data=None), IngestMode.OPTIONAL)
    assert route is EmbedRoute.TEXT and cap and "image 1" in cap


def test_plan_omni_mode():
    # bytes present -> omni regardless of alt-text
    route, cap = plan_embedding(_img(alt="ignored", data=PNG_1x1), IngestMode.OMNI)
    assert route is EmbedRoute.OMNI and cap is None
    # no bytes -> graceful text fallback (alt-text wins when present)
    route, cap = plan_embedding(_img(alt="desc", data=None), IngestMode.OMNI)
    assert route is EmbedRoute.TEXT and cap == "desc"
    route, cap = plan_embedding(_img(alt=None, data=None), IngestMode.OMNI)
    assert route is EmbedRoute.TEXT and "image 1" in cap


def test_build_extracted_images_local_and_inline(tmp_path):
    (tmp_path / "pics").mkdir()
    (tmp_path / "pics" / "real.png").write_bytes(PNG_1x1)
    b64 = base64.b64encode(PNG_1x1).decode()
    body = (
        "![local one](pics/real.png)\n"
        f"![inline](data:image/png;base64,{b64})\n"
        "![missing](pics/nope.png)\n"
        "![remote](https://example.com/x.png)\n"
    )
    imgs = build_extracted_images("doc", body, search_dirs=[tmp_path])
    assert len(imgs) == 4
    local, inline, missing, remote = imgs

    assert local.has_data and local.mime_type == "image/png"
    assert local.has_alt_text and local.alt_text == "local one"

    assert inline.has_data and inline.mime_type == "image/png"

    assert not missing.has_data            # file not found -> no bytes
    assert remote.data is None             # remote not fetched by default

    # ids are stable + unique per (concept, index, src)
    assert len({im.asset_id for im in imgs}) == 4


def test_okf_asset_resolves_from_store(tmp_path):
    # Bytes staged in the bundle's _assets store under <id>.<ext>
    assets = tmp_path / "_assets"
    assets.mkdir()
    (assets / "img_deadbeef.png").write_bytes(PNG_1x1)
    body = "![a diagram](okf-asset://img_deadbeef)\n![gone](okf-asset://img_missing)\n"
    imgs = build_extracted_images("doc", body, search_dirs=[tmp_path])
    assert len(imgs) == 2
    resolved, missing = imgs

    # Resolved: bytes present -> omni-eligible, id passed through verbatim
    assert resolved.asset_id == "img_deadbeef"
    assert resolved.has_data and resolved.mime_type == "image/png"
    route, cap = plan_embedding(resolved, IngestMode.OMNI)
    assert route is EmbedRoute.OMNI and cap is None

    # Missing on disk: no bytes -> graceful text fallback even in omni mode
    assert missing.asset_id == "img_missing"
    assert not missing.has_data
    route, cap = plan_embedding(missing, IngestMode.OMNI)
    assert route is EmbedRoute.TEXT


def run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    import inspect
    import tempfile
    passed = 0
    for fn in fns:
        params = inspect.signature(fn).parameters
        if "tmp_path" in params:
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d))
        else:
            fn()
        passed += 1
        print(f"  PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    run()
