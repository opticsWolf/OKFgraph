#!/usr/bin/env python3
"""
OKF Search Browser (PySide6)

A desktop search front-end for an okfgraph database. Pick a database, choose a
search mode, and browse results with a live preview pane that renders concept
markdown — including images pulled straight from the database BLOBs.

Search modes
------------
* Concepts — hybrid (semantic vector + full-text, RRF), with optional
  type / tags / parent-directory filters.
* Images — semantic text→image search over the unified vector index, with an
  option to encode the query through the omni text side.

All database and model work runs on a background thread, so the UI stays
responsive while the (first) query loads the embedding model. Image bytes are
resolved from the database (``get_image_data``) and rendered inline via a custom
``loadResource`` that understands ``okf-asset://<id>`` links — the same protocol
the converter/ingest tools write.

Run:  python okf_search_browser.py
"""

import re
import sys
import queue
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox, QCheckBox,
    QListWidget, QListWidgetItem, QTextBrowser, QSplitter, QFileDialog,
    QMessageBox, QFrame,
)
from PySide6.QtCore import Qt, QThread, Signal, QByteArray
from PySide6.QtGui import QImage, QTextDocument

# Resolve the ImageResource enum across PySide6 versions.
try:
    _IMAGE_RES = QTextDocument.ResourceType.ImageResource
except AttributeError:  # pragma: no cover - older binding layout
    _IMAGE_RES = QTextDocument.ImageResource

VALID_DIMS = [512, 256, 768, 1024, 128, 64, 32]
_ASSET_ID_RE = re.compile(r"okf-asset://([A-Za-z0-9_./:-]+)")


# ── Pure helpers (no Qt / no okfgraph — unit-testable) ──────────────────

def extract_asset_ids(text: str) -> List[str]:
    """Return the okf-asset ids referenced in *text*, in order, de-duplicated."""
    return list(dict.fromkeys(_ASSET_ID_RE.findall(text or "")))


def concept_result_label(r: Dict[str, Any]) -> str:
    score = r.get("relevance_score") or 0.0
    title = r.get("title") or r.get("id") or "(untitled)"
    ctype = r.get("type") or "?"
    return f"{score:0.3f}   {title}   ·   {ctype}"


def image_result_label(r: Dict[str, Any]) -> str:
    score = r.get("relevance_score") or 0.0
    label = r.get("alt_text") or r.get("file_name") or r.get("id") or "(image)"
    route = r.get("embed_route") or "?"
    return f"{score:0.3f}   {label}   ·   {route}"


def build_concept_markdown(payload: Dict[str, Any]) -> str:
    """Compose a markdown document for a concept preview (header + body)."""
    cid = payload.get("id") or ""
    if payload.get("missing"):
        return f"# Not found\n\nNo concept with id `{cid}` exists in this database."
    title = payload.get("title") or cid or "(untitled)"
    meta: List[str] = []
    if payload.get("type"):
        meta.append(f"**type:** {payload['type']}")
    tags = payload.get("tags") or []
    if tags:
        meta.append("**tags:** " + ", ".join(str(t) for t in tags))
    if cid:
        meta.append(f"**id:** `{cid}`")

    lines = [f"# {title}", ""]
    if meta:
        lines += ["  ·  ".join(meta), ""]
    desc = payload.get("description")
    if desc:
        lines += [f"> {desc}", ""]
    lines += ["---", "", payload.get("body") or "_(no body)_"]
    return "\n".join(lines)


def build_image_markdown(meta: Dict[str, Any], including: Optional[List[str]] = None) -> str:
    """Compose a markdown document for an image preview."""
    aid = meta.get("id") or ""
    alt = meta.get("alt_text") or ""
    heading = alt or meta.get("file_name") or aid or "Image"
    lines = [f"# {heading}", "", f"![{alt}](okf-asset://{aid})", ""]
    info: List[str] = []
    if meta.get("file_name"):
        info.append(f"**file:** {meta['file_name']}")
    if meta.get("mime_type"):
        info.append(f"**mime:** {meta['mime_type']}")
    if meta.get("embed_route"):
        info.append(f"**embedded via:** {meta['embed_route']}")
    if info:
        lines += ["  ·  ".join(info), ""]
    if aid:
        lines += [f"**id:** `{aid}`", ""]
    if alt:
        lines += [f"**alt text:** {alt}", ""]
    if including:
        lines += ["**included by:** " + ", ".join(f"`{c}`" for c in including)]
    return "\n".join(lines)


# ── Database backend (plain class — owns the router, all query logic) ────

class DbBackend:
    """Owns an OKFRouter and answers search/preview requests.

    Kept free of Qt so it can be unit-tested with a fake router. The router
    (and its embedding model) are built lazily and rebuilt when the config
    changes (e.g. the user picks a different database or dimension).
    """

    def __init__(self) -> None:
        self.router = None
        self._cfg: Optional[Dict[str, Any]] = None

    def needs_rebuild(self, cfg: Dict[str, Any]) -> bool:
        return self.router is None or cfg != self._cfg

    def ensure_router(self, cfg: Dict[str, Any]) -> None:
        if not self.needs_rebuild(cfg):
            return
        from okfgraph.router import OKFRouter  # lazy: heavy import

        # Release the previous handle first — Ladybug permits only one live
        # handle per database file, so a lingering open router (e.g. after the
        # user switches databases) would make the new open fail.
        if self.router is not None and hasattr(self.router, "close"):
            try:
                self.router.close()
            except Exception:
                pass
            self.router = None

        self.router = OKFRouter(
            db_path=cfg["db"],
            bundle_root=cfg["bundle"],
            embedding_dim=int(cfg["dim"]),
            device=cfg.get("device", "cpu"),
        )
        self._cfg = dict(cfg)

    # -- searches --------------------------------------------------------

    def search_concepts(
        self, query: str, ctype: str, tags: List[str], parent: str, limit: int
    ) -> List[Dict[str, Any]]:
        results = self.router.search_hybrid(
            query=query,
            concept_type=ctype or None,
            tags=tags or None,
            parent_id=parent or None,
            limit=limit,
        )
        for r in results:
            r["_kind"] = "concept"
        return results

    def search_images(self, query: str, use_omni: bool, limit: int) -> List[Dict[str, Any]]:
        results = self.router.search_images_with_text(
            text_query=query, use_text_model=not use_omni, limit=limit
        )
        for r in results:
            r["_kind"] = "image"
        return results

    # -- previews (resolve image BLOBs here, on the worker thread) --------

    def preview_concept(self, concept_id: str) -> Dict[str, Any]:
        concept = self.router.get_by_id(concept_id)
        if concept is None:
            return {"_kind": "concept", "id": concept_id, "missing": True, "images": {}}
        body = getattr(concept, "body", "") or ""
        images: Dict[str, bytes] = {}
        for aid in extract_asset_ids(body):
            rec = self.router.get_image_data(aid)
            if rec and rec.get("data"):
                images[aid] = bytes(rec["data"])
        return {
            "_kind": "concept",
            "id": getattr(concept, "id", concept_id),
            "title": getattr(concept, "title", None),
            "type": getattr(concept, "type", None),
            "tags": list(getattr(concept, "tags", None) or []),
            "description": getattr(concept, "description", None),
            "body": body,
            "images": images,
        }

    def preview_image(self, asset_id: str) -> Dict[str, Any]:
        rec = self.router.get_image_data(asset_id) or {"id": asset_id}
        data = bytes(rec["data"]) if rec.get("data") else b""
        meta = {k: rec.get(k) for k in ("id", "file_name", "mime_type", "alt_text", "embed_route")}
        return {
            "_kind": "image",
            "meta": meta,
            "data": data,
            "including": self._including_concepts(asset_id),
        }

    def _including_concepts(self, asset_id: str) -> List[str]:
        try:
            res = self.router.conn.execute(
                "MATCH (c:Concept)-[:INCLUDES_ASSET]->(i:ImageAsset {id: $iid}) "
                "RETURN c.id AS id",
                {"iid": asset_id},
            )
            return [row["id"] for row in res.rows_as_dict().get_all()]
        except Exception:
            return []


# ── Worker thread: serialises all DB/model access through a queue ───────

class DbWorker(QThread):
    status = Signal(str)
    error = Signal(str)
    concept_results = Signal(object)
    image_results = Signal(object)
    preview_ready = Signal(object)

    def __init__(self, backend: DbBackend):
        super().__init__()
        self.backend = backend
        self._q: "queue.Queue" = queue.Queue()

    def submit(self, kind: str, payload: Dict[str, Any]) -> None:
        self._q.put((kind, payload))

    def stop(self) -> None:
        self._q.put(None)
        self.wait(3000)

    def run(self) -> None:
        while True:
            job = self._q.get()
            if job is None:
                break
            kind, payload = job
            cfg = payload.get("cfg")
            try:
                if cfg and self.backend.needs_rebuild(cfg):
                    self.status.emit("Opening database and loading model…")
                if cfg:
                    self.backend.ensure_router(cfg)

                if kind == "search_concepts":
                    self.concept_results.emit(self.backend.search_concepts(**payload["args"]))
                elif kind == "search_images":
                    self.image_results.emit(self.backend.search_images(**payload["args"]))
                elif kind == "preview_concept":
                    self.preview_ready.emit(self.backend.preview_concept(payload["id"]))
                elif kind == "preview_image":
                    self.preview_ready.emit(self.backend.preview_image(payload["id"]))
            except Exception as e:  # surface failures without killing the thread
                self.error.emit(f"{kind.replace('_', ' ')} failed: {e}")


# ── Preview pane: renders markdown and resolves okf-asset:// images ─────

class PreviewBrowser(QTextBrowser):
    """QTextBrowser that resolves ``okf-asset://<id>`` images from an in-memory
    byte map (populated per preview from the database)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setOpenExternalLinks(True)
        self._image_map: Dict[str, bytes] = {}

    def render_markdown(self, md: str, image_map: Optional[Dict[str, bytes]] = None) -> None:
        self._image_map = image_map or {}
        self.setMarkdown(md)

    def loadResource(self, resource_type: int, url) -> Any:
        try:
            key = url.toString() if hasattr(url, "toString") else str(url)
        except Exception:
            key = str(url)
        aid = key
        if aid.startswith("okf-asset://"):
            aid = aid[len("okf-asset://"):]
        aid = aid.strip().strip("/")
        data = self._image_map.get(aid) or self._image_map.get(key)
        if data:
            img = QImage()
            img.loadFromData(QByteArray(data))
            return img
        if resource_type == _IMAGE_RES:
            # Don't let the preview reach out to the network/filesystem for
            # anything other than database-backed assets.
            return QImage()
        return super().loadResource(resource_type, url)


# ── Main window ─────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OKF Search Browser")
        self.resize(1000, 640)

        self.backend = DbBackend()
        self.worker = DbWorker(self.backend)
        self.worker.status.connect(self._set_status)
        self.worker.error.connect(self._on_error)
        self.worker.concept_results.connect(self._on_concept_results)
        self.worker.image_results.connect(self._on_image_results)
        self.worker.preview_ready.connect(self._on_preview_ready)
        self.worker.start()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # -- config row: database + dimension -----------------------------
        cfg_row = QHBoxLayout()
        cfg_row.addWidget(QLabel("Database:"))
        self.db_edit = QLineEdit("okfgraph.db")
        cfg_row.addWidget(self.db_edit, stretch=1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_db)
        cfg_row.addWidget(browse)
        cfg_row.addWidget(QLabel("Dim:"))
        self.dim_combo = QComboBox()
        self.dim_combo.addItems([str(d) for d in VALID_DIMS])
        cfg_row.addWidget(self.dim_combo)
        root.addLayout(cfg_row)

        # -- search row: mode + query + go --------------------------------
        search_row = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Concepts (hybrid)", "Images (semantic)"])
        self.mode_combo.currentIndexChanged.connect(self._sync_filter_widgets)
        search_row.addWidget(self.mode_combo)

        self.query_edit = QLineEdit()
        self.query_edit.setPlaceholderText("Type a query and press Enter…")
        self.query_edit.returnPressed.connect(self._do_search)
        search_row.addWidget(self.query_edit, stretch=1)

        self.search_btn = QPushButton("Search")
        self.search_btn.clicked.connect(self._do_search)
        search_row.addWidget(self.search_btn)
        root.addLayout(search_row)

        # -- filter row: concept filters + image option + limit -----------
        filt = QGridLayout()
        filt.setHorizontalSpacing(8)
        self.type_edit = QLineEdit(); self.type_edit.setPlaceholderText("type")
        self.tags_edit = QLineEdit(); self.tags_edit.setPlaceholderText("tags (comma-separated)")
        self.parent_edit = QLineEdit(); self.parent_edit.setPlaceholderText("parent directory id")
        self.omni_cb = QCheckBox("Query with omni model")
        self.limit_spin = QSpinBox(); self.limit_spin.setRange(1, 100); self.limit_spin.setValue(10)

        filt.addWidget(QLabel("Filters:"), 0, 0)
        filt.addWidget(self.type_edit, 0, 1)
        filt.addWidget(self.tags_edit, 0, 2)
        filt.addWidget(self.parent_edit, 0, 3)
        filt.addWidget(self.omni_cb, 0, 4)
        filt.addWidget(QLabel("Limit:"), 0, 5)
        filt.addWidget(self.limit_spin, 0, 6)
        filt.setColumnStretch(1, 1)
        filt.setColumnStretch(2, 2)
        filt.setColumnStretch(3, 2)
        root.addLayout(filt)

        # -- results | preview --------------------------------------------
        splitter = QSplitter(Qt.Horizontal)
        self.results = QListWidget()
        self.results.itemSelectionChanged.connect(self._on_result_selected)
        self.results.setMinimumWidth(300)
        splitter.addWidget(self.results)

        self.preview = PreviewBrowser()
        splitter.addWidget(self.preview)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 660])
        root.addWidget(splitter, stretch=1)

        # -- status -------------------------------------------------------
        self.status_lbl = QLabel("Ready.")
        self.status_lbl.setStyleSheet("color: #7f8c8d;")
        root.addWidget(self.status_lbl)

        self._sync_filter_widgets()

    # -- config -----------------------------------------------------------

    def _cfg(self) -> Dict[str, Any]:
        db = self.db_edit.text().strip() or "okfgraph.db"
        bundle = str(Path(db).resolve().parent)
        return {"db": db, "bundle": bundle, "dim": int(self.dim_combo.currentText()), "device": "cpu"}

    def _browse_db(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select okfgraph database", "", "LadybugDB (*.db);;All Files (*)"
        )
        if path:
            self.db_edit.setText(path)

    def _is_concept_mode(self) -> bool:
        return self.mode_combo.currentIndex() == 0

    def _sync_filter_widgets(self):
        concept = self._is_concept_mode()
        for w in (self.type_edit, self.tags_edit, self.parent_edit):
            w.setEnabled(concept)
        self.omni_cb.setEnabled(not concept)

    # -- search -----------------------------------------------------------

    def _do_search(self):
        query = self.query_edit.text().strip()
        if not query:
            self._set_status("Enter a query first.")
            return
        cfg = self._cfg()
        self.search_btn.setEnabled(False)
        self.results.clear()
        self.preview.clear()
        self._set_status("Searching…")
        if self._is_concept_mode():
            tags = [t.strip() for t in self.tags_edit.text().split(",") if t.strip()]
            self.worker.submit("search_concepts", {
                "cfg": cfg,
                "args": {
                    "query": query,
                    "ctype": self.type_edit.text().strip(),
                    "tags": tags,
                    "parent": self.parent_edit.text().strip(),
                    "limit": self.limit_spin.value(),
                },
            })
        else:
            self.worker.submit("search_images", {
                "cfg": cfg,
                "args": {
                    "query": query,
                    "use_omni": self.omni_cb.isChecked(),
                    "limit": self.limit_spin.value(),
                },
            })

    def _populate(self, results: List[Dict[str, Any]], labeller) -> None:
        self.search_btn.setEnabled(True)
        self.results.clear()
        for r in results:
            item = QListWidgetItem(labeller(r))
            item.setData(Qt.UserRole, r)
            self.results.addItem(item)
        self._set_status(f"{len(results)} result(s).")
        if results:
            self.results.setCurrentRow(0)

    def _on_concept_results(self, results):
        self._populate(results, concept_result_label)

    def _on_image_results(self, results):
        self._populate(results, image_result_label)

    # -- preview ----------------------------------------------------------

    def _on_result_selected(self):
        item = self.results.currentItem()
        if item is None:
            return
        r = item.data(Qt.UserRole) or {}
        cfg = self._cfg()
        if r.get("_kind") == "image":
            self.worker.submit("preview_image", {"cfg": cfg, "id": r.get("id")})
        else:
            self.worker.submit("preview_concept", {"cfg": cfg, "id": r.get("id")})

    def _on_preview_ready(self, payload: Dict[str, Any]):
        if payload.get("_kind") == "image":
            meta = payload.get("meta") or {}
            md = build_image_markdown(meta, payload.get("including"))
            image_map = {meta.get("id"): payload.get("data") or b""}
            self.preview.render_markdown(md, image_map)
        else:
            md = build_concept_markdown(payload)
            self.preview.render_markdown(md, payload.get("images") or {})

    # -- misc -------------------------------------------------------------

    def _set_status(self, msg: str):
        if msg:
            self.status_lbl.setText(msg)

    def _on_error(self, msg: str):
        self.search_btn.setEnabled(True)
        self._set_status(msg)
        QMessageBox.warning(self, "Error", msg)

    def closeEvent(self, event):
        try:
            self.worker.stop()
        except Exception:
            pass
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
