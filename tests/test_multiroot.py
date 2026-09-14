"""Phase 2 §2.1 — multi-root identity (0.4.0).

Validation + parse-alias parts need no encoder. Import parts use the real
wheel on tiny docs (repo convention: no stub encoders).
"""

import logging

import pytest

from okfgraph.components.import_ import parse_source_file
from okfgraph.components.roots import (
    namespaced_id,
    prefix_key,
    resolve_alias_for_path,
    split_namespaced,
    strip_prefix,
    validate_roots,
)


def _mkroot(tmp_path, name, files):
    root = tmp_path / name
    root.mkdir()
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return root


def _doc(title):
    return f"---\ntitle: {title}\ntype: note\n---\n\nBody of {title}.\n"


class TestRootsValidation:
    def test_none_and_empty(self, tmp_path):
        assert validate_roots(None, primary=str(tmp_path)) == {}
        assert validate_roots({}, primary=str(tmp_path)) == {}

    def test_ok_returns_resolved(self, tmp_path):
        prim = _mkroot(tmp_path, "prim", {})
        a = _mkroot(tmp_path, "a", {})
        b = _mkroot(tmp_path, "b", {})
        out = validate_roots({"aa": str(a), "bb": str(b)}, primary=str(prim))
        assert sorted(out) == ["aa", "bb"]
        assert all(p.is_absolute() for p in out.values())

    @pytest.mark.parametrize("alias", ["@x", "a:b", "a/b", "a b", "", 3])
    def test_bad_alias_rejected(self, tmp_path, alias):
        a = _mkroot(tmp_path, "a", {})
        with pytest.raises(ValueError, match="[Aa]lias"):
            validate_roots({alias: str(a)}, primary=str(tmp_path))

    def test_overlap_rejected(self, tmp_path):
        a = _mkroot(tmp_path, "a", {})
        (a / "sub").mkdir()
        with pytest.raises(ValueError, match="overlap"):
            validate_roots({"aa": str(a), "nest": str(a / "sub")},
                           primary=str(tmp_path))
        with pytest.raises(ValueError, match="overlap"):
            validate_roots({"aa": str(a), "dup": str(a)}, primary=str(tmp_path))
        # Primary tree participates in the overlap check ("" root).
        with pytest.raises(ValueError, match="overlap"):
            validate_roots({"aa": str(tmp_path)}, primary=str(a))

    def test_resolve_longest_prefix_and_outside(self, tmp_path):
        a = _mkroot(tmp_path, "a", {})
        (a / "sub").mkdir()
        prim = _mkroot(tmp_path, "prim", {})
        roots = validate_roots({"aa": str(a)}, primary=str(prim))
        assert resolve_alias_for_path(a / "sub" / "x.md", roots) == "aa"
        assert resolve_alias_for_path(prim / "z.md", roots) is None
        assert resolve_alias_for_path(a, roots) == "aa"

    def test_prefix_math(self):
        assert prefix_key("aa", "sub/f.md") == "@aa/sub/f.md"
        assert prefix_key("aa", "@aa/sub/f.md") == "@aa/sub/f.md"  # idempotent
        assert prefix_key("", "f.md") == "f.md"
        assert strip_prefix("aa", "@aa/sub/f.md") == "sub/f.md"
        assert strip_prefix("aa", "@bb/x.md") is None
        assert strip_prefix("aa", "f.md") is None
        assert strip_prefix("", "@aa/x.md") == "@aa/x.md"  # legacy owns all
        assert namespaced_id("aa", "d") == "@aa/d"
        assert namespaced_id("", "d") == "d"
        assert split_namespaced("@aa/b/c") == ("aa", "b/c")
        assert split_namespaced("plain") == ("", "plain")


class TestParseAlias:
    def test_namespaced_and_bare(self, tmp_path):
        root = _mkroot(tmp_path, "r", {"sub/doc.md": _doc("D")})
        _, _, cid = parse_source_file(root / "sub" / "doc.md", root, "aa")
        assert cid == "@aa/sub/doc"
        _, _, bare = parse_source_file(root / "sub" / "doc.md", root)
        assert bare == "sub/doc"

    def test_outside_stays_bare(self, tmp_path):
        root = _mkroot(tmp_path, "r", {})
        other = _mkroot(tmp_path, "o", {"x.md": _doc("X")})
        _, _, cid = parse_source_file(other / "x.md", root, "aa")
        assert cid == "x"

    def test_frontmatter_uid_preserved(self, tmp_path):
        root = _mkroot(tmp_path, "r", {
            "d.md": "---\ntitle: D\ntype: note\nid: stable-1\n---\n\nB.\n"})
        concept, _, cid = parse_source_file(root / "d.md", root, "aa")
        assert cid == "@aa/d"
        assert (concept.model_extra or {}).get("uid") == "stable-1"


def _mrouter(tmp_path, prim, roots):
    from okfgraph.router import OKFRouter

    return OKFRouter(
        db_path=str(tmp_path / "m.db"), bundle_root=str(prim), roots=roots,
        embedding_dim=512, enable_chunking=False, device="cpu")


class TestMultiRootImport:
    def test_collision_free_ids(self, tmp_path):
        prim = _mkroot(tmp_path, "prim", {"README.md": _doc("Prim")})
        a = _mkroot(tmp_path, "a", {"README.md": _doc("Alpha")})
        b = _mkroot(tmp_path, "b", {"sub/README.md": _doc("Beta")})
        r = _mrouter(tmp_path, prim, {"aa": str(a), "bb": str(b)})
        try:
            ids = r.import_mgr.import_bundle()
            assert sorted(ids) == ["@aa/README", "@bb/sub/README", "README"]
            rows = r.conn.execute(
                "MATCH (c:Concept) RETURN c.id AS id ORDER BY c.id"
            ).rows_as_dict().get_all()
            assert [x["id"] for x in rows] == [
                "@aa/README", "@bb/sub/README", "README"]
            # Second run: delta silent everywhere.
            assert r.import_mgr.import_bundle() == []
        finally:
            r.close()

    def test_delta_isolation_and_purge(self, tmp_path):
        prim = _mkroot(tmp_path, "prim", {})
        a = _mkroot(tmp_path, "a", {"one.md": _doc("One")})
        b = _mkroot(tmp_path, "b", {"two.md": _doc("Two")})
        r = _mrouter(tmp_path, prim, {"aa": str(a), "bb": str(b)})
        try:
            assert sorted(r.import_mgr.import_bundle()) == ["@aa/one", "@bb/two"]
            # Touch one root only → only its concept re-encodes.
            (a / "one.md").write_text(_doc("One v2"), encoding="utf-8")
            assert r.import_mgr.import_bundle() == ["@aa/one"]
            # Delete from the other root + purge → only it goes.
            (b / "two.md").unlink()
            assert r.import_mgr.import_bundle(purge_deleted=True) == []
            remaining = r.conn.execute(
                "MATCH (c:Concept) RETURN c.id AS id"
            ).rows_as_dict().get_all()
            assert [x["id"] for x in remaining] == ["@aa/one"]
            # Tombstone key is namespaced; FileHash rows are namespaced.
            hashes = r.conn.execute(
                "MATCH (f:FileHash) RETURN f.path AS p, f.concept_id AS c "
                "ORDER BY f.path"
            ).rows_as_dict().get_all()
            assert [(h["p"], h["c"]) for h in hashes] == [
                ("@aa/one.md", "@aa/one")]
        finally:
            r.close()

    def test_per_alias_directory_nodes(self, tmp_path):
        prim = _mkroot(tmp_path, "prim", {})
        a = _mkroot(tmp_path, "a", {"d.md": _doc("D")})
        r = _mrouter(tmp_path, prim, {"aa": str(a)})
        try:
            r.import_mgr.import_bundle()
            rows = r.conn.execute(
                "MATCH (d:Directory) RETURN d.id AS id ORDER BY d.id"
            ).rows_as_dict().get_all()
            assert "@aa" in [x["id"] for x in rows]
        finally:
            r.close()

class TestLiveness:
    def test_absent_root_skipped_no_tombstones(self, tmp_path, caplog):
        prim = _mkroot(tmp_path, "prim", {})
        a = _mkroot(tmp_path, "a", {"one.md": _doc("One")})
        missing = tmp_path / "gone"  # never created: unmounted root
        r = _mrouter(tmp_path, prim, {"aa": str(a), "bb": str(missing)})
        try:
            with caplog.at_level(logging.WARNING,
                                  logger="okfgraph.components.import_"):
                assert r.import_mgr.import_bundle() == ["@aa/one"]
            assert any("not present" in m for m in caplog.messages)
            assert r.conn.execute(
                "MATCH (d:DeletedPath) RETURN count(d) AS n"
            ).rows_as_dict().get_all()[0]["n"] == 0
            paths = [x["p"] for x in r.conn.execute(
                "MATCH (f:FileHash) RETURN f.path AS p"
            ).rows_as_dict().get_all()]
            assert not [p for p in paths if p.startswith("@bb/")]
        finally:
            r.close()

    def test_purge_refused_with_absent_root(self, tmp_path):
        prim = _mkroot(tmp_path, "prim", {})
        a = _mkroot(tmp_path, "a", {"one.md": _doc("One")})
        r = _mrouter(tmp_path, prim,
                      {"aa": str(a), "bb": str(tmp_path / "gone")})
        try:
            r.import_mgr.import_bundle()
            with pytest.raises(RuntimeError, match="purge refused"):
                r.import_mgr.import_bundle(purge_deleted=True)
            # Refusal happens before any consumption: concept intact.
            assert [x["id"] for x in r.conn.execute(
                "MATCH (c:Concept) RETURN c.id AS id"
            ).rows_as_dict().get_all()] == ["@aa/one"]
        finally:
            r.close()

    def test_remount_resumes(self, tmp_path):
        prim = _mkroot(tmp_path, "prim", {})
        a = _mkroot(tmp_path, "a", {"one.md": _doc("One")})
        bdir = tmp_path / "b"
        r = _mrouter(tmp_path, prim, {"aa": str(a), "bb": str(bdir)})
        try:
            assert r.import_mgr.import_bundle() == ["@aa/one"]
            bdir.mkdir()
            (bdir / "two.md").write_text(_doc("Two"), encoding="utf-8")
            assert r.import_mgr.import_bundle() == ["@bb/two"]
        finally:
            r.close()

    def test_detach_multi_provenance_and_reattach(self, tmp_path):
        prim = _mkroot(tmp_path, "prim", {"p.md": _doc("P")})
        a = _mkroot(tmp_path, "a", {"q.md": _doc("Q")})
        r = _mrouter(tmp_path, prim, {"aa": str(a)})
        try:
            assert sorted(r.import_mgr.import_bundle()) == ["@aa/q", "p"]
            report = r.import_mgr.detach()
            assert report["verified"] is True
            assert report["file_count"] == 2
            prov = {row["alias"]: row for row in report["provenance"]}
            assert sorted(prov) == ["", "aa"]
            assert prov["aa"]["file_count"] == 1
            with pytest.raises(RuntimeError, match="detached"):
                r.import_mgr.import_bundle()
            # Full-tree --force re-attaches (root-set match). The baseline
            # was dropped at detach, so files re-encode — but upsert means
            # no duplicates, and a follow-up run is delta-silent.
            assert sorted(r.import_mgr.import_bundle(force=True)) == ["@aa/q", "p"]
            assert r.import_mgr.delta_mgr.is_detached() is False
            assert r.import_mgr.import_bundle() == []
            assert len(r.conn.execute(
                "MATCH (c:Concept) RETURN c.id AS id"
            ).rows_as_dict().get_all()) == 2
        finally:
            r.close()

    def test_detach_absent_root_refuses_verify(self, tmp_path):
        prim = _mkroot(tmp_path, "prim", {"p.md": _doc("P")})
        r = _mrouter(tmp_path, prim, {"bb": str(tmp_path / "gone")})
        try:
            r.import_mgr.import_bundle()
            with pytest.raises(RuntimeError, match="not found"):
                r.import_mgr.detach()
            report = r.import_mgr.detach(verify=False)
            assert report["verified"] is False
            assert len(report["provenance"]) == 2
        finally:
            r.close()

class TestCrossRootLinks:
    def _linked(self, tmp_path):
        prim = _mkroot(tmp_path, "prim", {})
        a = _mkroot(tmp_path, "a", {"target.md": _doc("Shared Target")})
        b = _mkroot(tmp_path, "b", {
            "target.md": _doc("Shared Target"),
            "hub.md": (
                "---\ntitle: Hub\ntype: note\n---\n\n"
                "See [[aa/target]] and [[@bb/target]] and [[AA/target]] "
                "and [[target]] and [[AA/TARGET]].\n"),
        })
        return _mrouter(tmp_path, prim, {"aa": str(a), "bb": str(b)})

    def test_qualified_links_resolve(self, tmp_path):
        r = self._linked(tmp_path)
        try:
            r.import_mgr.import_bundle()
            edges = sorted(
                x["dst"] for x in r.conn.execute(
                    "MATCH (a:Concept {id: '@bb/hub'})-[:LINKS_TO]->(b:Concept) "
                    "RETURN b.id AS dst"
                ).rows_as_dict().get_all())
            # alias/rest, @alias/rest, and case-folded alias all resolve;
            # the rest segment stays exact. (Edges dedupe on
            # source→target, so the two @aa/target links collapse to one;
            # the BrokenLink test below proves the case-fold resolved.)
            assert edges == ["@aa/target", "@bb/target"]
        finally:
            r.close()

    def test_unqualified_collision_breaks(self, tmp_path):
        r = self._linked(tmp_path)
        try:
            r.import_mgr.import_bundle()
            broken = sorted(
                x["t"] for x in r.conn.execute(
                    "MATCH (bl:BrokenLink {source_id: '@bb/hub'}) "
                    "RETURN bl.target_id AS t"
                ).rows_as_dict().get_all())
            # Ambiguous 'target' (two roots) + wrong-case rest never resolve.
            assert broken == ["AA/TARGET", "target"]
        finally:
            r.close()

class TestSurfaces:
    def test_cli_bundle_root_flag(self):
        from okfgraph.cli import build_parser

        args = build_parser().parse_args(
            ["import", "--all", "--bundle-root", "aa=/tmp/a",
             "--bundle-root", "bb=/tmp/b"])
        assert args.bundle_root == ["aa=/tmp/a", "bb=/tmp/b"]
        args = build_parser().parse_args(["import", "--all"])
        assert args.bundle_root is None

    def test_cli_roots_reach_router(self, tmp_path):
        from okfgraph.cli import _router
        from okfgraph.cli import build_parser

        prim = _mkroot(tmp_path, "prim", {})
        a = _mkroot(tmp_path, "a", {})
        args = build_parser().parse_args(
            ["doctor", "--db", str(tmp_path / "m.db"),
             "--bundle", str(prim), "--bundle-root", f"aa={a}"])
        router = _router(args)
        try:
            assert sorted(router.roots) == ["aa"]
            assert router.import_mgr.roots["aa"] == a.resolve()
        finally:
            router.close()

    def test_cli_bad_bundle_root_is_clean_error(self, tmp_path, capsys):
        from okfgraph.cli import _router
        from okfgraph.cli import build_parser

        args = build_parser().parse_args(
            ["doctor", "--bundle-root", "no-equals-here"])
        with pytest.raises(SystemExit) as e:
            _router(args)
        assert e.value.code == 2
        assert "ALIAS=PATH" in capsys.readouterr().out

    def test_toml_roots_relative_to_toml(self, tmp_path):
        from okfgraph.config import OKFConfig

        docs = tmp_path / "docs"
        docs.mkdir()
        (tmp_path / "okfgraph.toml").write_text(
            '[[roots]]\nalias = "aa"\npath = "docs"\n',
            encoding="utf-8")
        cfg = OKFConfig.load(bundle_root=tmp_path)
        assert cfg.roots == {"aa": str(docs)}
        # CLI list replaces TOML wholesale.
        cfg2 = OKFConfig.load(
            bundle_root=tmp_path, cli_args={"roots": ["bb=/x"]})
        assert cfg2.roots == {"bb": "/x"}
        with pytest.raises(ValueError, match="ALIAS=PATH"):
            OKFConfig.load(bundle_root=tmp_path,
                           cli_args={"roots": ["broken"]})

    def test_mcp_roots_plumbing(self, tmp_path):
        from okfgraph.mcp_server import create_mcp_server

        a = _mkroot(tmp_path, "a", {})
        mcp = create_mcp_server(
            db_path=str(tmp_path / "m.db"),
            bundle_root=str(tmp_path),
            roots={"aa": str(a)},
        )
        assert mcp is not None

    def test_ingest_md_namespaced(self, tmp_path):
        prim = _mkroot(tmp_path, "prim", {})
        a = _mkroot(tmp_path, "a", {"inside.md": _doc("Inside")})
        outside = tmp_path / "lone.md"
        outside.write_text(_doc("Lone"), encoding="utf-8")
        r = _mrouter(tmp_path, prim, {"aa": str(a)})
        try:
            got_in = r.ingest_mgr.ingest_md(a / "inside.md")
            assert got_in["concept_id"] == "@aa/inside"
            got_out = r.ingest_mgr.ingest_md(outside)
            assert got_out["concept_id"] == "lone"
        finally:
            r.close()

    def test_pdf_namespace_mechanics(self, tmp_path):
        """import_bundle(alias=...) with a registered namespace detector
        mints stable namespaced IDs (the _import_work_dir shape)."""
        from okfgraph.components.delta import DeltaDetector

        prim = _mkroot(tmp_path, "prim", {})
        work = _mkroot(tmp_path, "work", {"page-1.md": _doc("P1")})
        r = _mrouter(tmp_path, prim, {})
        try:
            ns = "pdf-abc123def456"
            r.import_mgr._delta_by_alias[ns] = DeltaDetector(
                r.conn, work, ns)
            assert r.import_mgr.import_bundle(work, alias=ns) == [
                f"@{ns}/page-1"]
        finally:
            r.close()

    def test_export_round_trip(self, tmp_path):
        prim = _mkroot(tmp_path, "prim", {"p.md": _doc("P")})
        a = _mkroot(tmp_path, "a", {"sub/q.md": _doc("Q")})
        r = _mrouter(tmp_path, prim, {"aa": str(a)})
        try:
            assert sorted(r.import_mgr.import_bundle()) == ["@aa/sub/q", "p"]
            out = tmp_path / "export"
            exported = r.export_mgr.export_bundle(output_dir=out)
            assert sorted(exported) == ["@aa/sub/q", "p"]
            assert (out / "@aa" / "sub" / "q.md").is_file()
            assert (out / "p.md").is_file()
        finally:
            r.close()
        # Single-root re-import of the export tree reproduces exact IDs.
        (tmp_path / "db2").mkdir(exist_ok=True)
        r2 = _mrouter(tmp_path / "db2", out, {})
        try:
            assert sorted(r2.import_mgr.import_bundle()) == ["@aa/sub/q", "p"]
        finally:
            r2.close()

    def test_drift_multi_root(self, tmp_path):
        prim = _mkroot(tmp_path, "prim", {"p.md": _doc("P")})
        a = _mkroot(tmp_path, "a", {"q.md": _doc("Q")})
        r = _mrouter(tmp_path, prim, {"aa": str(a)})
        try:
            r.import_mgr.import_bundle()
            assert r.diff_db_dir(None)["identical"] is True
            (a / "q.md").write_text(_doc("Q v2"), encoding="utf-8")
            result = r.diff_db_dir(None)
            assert result["identical"] is False
            assert result["changed"] == ["@aa/q"]
        finally:
            r.close()

    def test_doctor_reports_roots(self, tmp_path):
        prim = _mkroot(tmp_path, "prim", {})
        a = _mkroot(tmp_path, "a", {"one.md": _doc("One")})
        r = _mrouter(tmp_path, prim, {"aa": str(a), "bb": str(tmp_path / "gone")})
        try:
            r.import_mgr.import_bundle()
            roots = {x["alias"]: x for x in r.diagnose()["roots"]}
            assert roots["aa"]["present"] is True
            assert roots["aa"]["concepts"] == 1
            assert roots["aa"]["tracked_files"] == 1
            assert roots["bb"]["present"] is False
        finally:
            r.close()

    def test_reserved_prefix_warns_but_imports(self, tmp_path, caplog):
        prim = _mkroot(tmp_path, "prim", {"@x/y.md": _doc("Why")})
        r = _mrouter(tmp_path, prim, {})
        try:
            with caplog.at_level(logging.WARNING, logger="okfgraph.components.import_"):
                assert r.import_mgr.import_bundle() == ["@x/y"]
            assert any("top-level '@'-prefixed" in m for m in caplog.messages)
        finally:
            r.close()
