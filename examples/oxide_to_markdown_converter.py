#!/usr/bin/env python3
"""
Document-to-Markdown Converter (PySide6 + pdf_oxide + office_oxide + okfgraph)
Drop PDF, DOCX, XLSX, PPTX, DOC, XLS, PPT files onto the window.
Output: <filename>.md in the same folder as the source.

When "Extract images" is enabled, embedded images are written to a sibling
``_assets/<id>.<ext>`` store and their markdown links are rewritten to
``![alt](okf-asset://<id>)``. This is the okfgraph ingestion contract: the
converter only extracts bytes and references them; okfgraph resolves those
links at import time and owns all embedding, BLOB storage, and graph linking.
No base64 is ever inlined.

When "Import into okfgraph" is enabled, each converted .md file is fed
directly into the okfgraph knowledge graph via OKFRouter (in-process, shared
embedding model). This skips the separate ``okf import`` CLI step.
"""

import sys
import re
import shutil
import hashlib
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QLabel, QTextBrowser, QWidget, QFileDialog, QMessageBox,
    QPushButton, QProgressBar, QFrame, QCheckBox, QGroupBox,
    QComboBox, QLineEdit, QFormLayout
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent

# Optional dependencies – show clear messages if missing
try:
    from pdf_oxide import PdfDocument
except ImportError:
    PdfDocument = None

try:
    from office_oxide import Document as OfficeDocument
except ImportError:
    OfficeDocument = None

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from okfgraph.router import OKFRouter
        from okfgraph.images import IngestMode
except ImportError:
    OKFRouter = None
    IngestMode = None

SUPPORTED_EXTS = {".pdf", ".docx", ".xlsx", ".pptx", ".doc", ".xls", ".ppt"}

# ── okf-asset:// staging convention ─────────────────────────────────────
# These must match okfgraph.images (ASSET_STORE_DIRNAME + the okf-asset://
# scheme). okfgraph resolves okf-asset://<id> from <bundle>/_assets/<id>.<ext>
# at import time, so the converter never embeds or touches a database.
ASSET_STORE_DIRNAME = "_assets"

# ``![alt](src)`` — src may carry an optional "title" we strip off.
_IMG_LINK_RE = re.compile(r'!\[(?P<alt>.*?)\]\((?P<src>.*?)\)')


def _split_src_and_title(raw: str) -> str:
    """Return just the path/URL part of a markdown image target."""
    raw = raw.strip()
    if not raw:
        return raw
    m = re.match(r'^(<[^>]+>|\S+)', raw)
    part = m.group(1) if m else raw
    if part.startswith("<") and part.endswith(">"):
        part = part[1:-1]
    return part


def _asset_id(concept_stem: str, occurrence: int, img_bytes: bytes) -> str:
    """Deterministic, concept-scoped asset id (matches okf_ingest_tool).

    Scoping by the owning document keeps ids unique per concept (so okfgraph's
    per-asset delete-then-create never clobbers an asset shared across
    concepts); hashing the bytes keeps re-runs on unchanged input idempotent.
    """
    h = hashlib.sha256()
    h.update(f"{concept_stem}|{occurrence}|".encode("utf-8"))
    h.update(img_bytes)
    return f"img_{h.hexdigest()[:16]}"


def _stage_images_as_okf_assets(
    md: str, image_src_dir: Path, source_path: Path, out_dir: Path, concept_stem: str
) -> tuple[str, int]:
    """Move extracted images into ``<out_dir>/_assets/<id>.<ext>`` and rewrite
    their links to ``okf-asset://<id>``. Returns ``(rewritten_md, count)``.

    Does no embedding or database work — that is okfgraph's job at ingest time.
    """
    assets_dir = out_dir / ASSET_STORE_DIRNAME
    occurrence = {"n": 0}

    def _repl(match: "re.Match") -> str:
        alt = match.group("alt").strip()
        src = _split_src_and_title(match.group("src"))
        low = src.lower()
        if (not src
                or low.startswith(("http://", "https://", "okf-asset://", "data:"))
                or src.startswith(f"{ASSET_STORE_DIRNAME}/")):
            return match.group(0)

        # Resolve the physical file: extractor temp dir, then doc-relative.
        cand = image_src_dir / Path(src).name
        if not cand.is_file():
            cand = source_path.parent / src
        if not cand.is_file() and Path(src).is_absolute():
            cand = Path(src)
        if not cand.is_file():
            return match.group(0)  # unresolved — leave the link untouched

        occurrence["n"] += 1
        data = cand.read_bytes()
        asset_id = _asset_id(concept_stem, occurrence["n"], data)
        suffix = cand.suffix.lower() or ".bin"
        assets_dir.mkdir(parents=True, exist_ok=True)
        dest = assets_dir / f"{asset_id}{suffix}"
        if not dest.exists():
            shutil.copy2(cand, dest)
        return f"![{alt}](okf-asset://{asset_id})"

    new_md = _IMG_LINK_RE.sub(_repl, md)
    return new_md, occurrence["n"]


class ConversionWorker(QThread):
    """Background thread – keeps the UI responsive while files are processed."""
    log_signal = Signal(str)
    progress_signal = Signal(int, int)   # current, total
    finished_signal = Signal(int, int, int)   # success_count, fail_count, skipped_count

    def __init__(self, files, seen_hashes, extract_images=True,
                 ingest_into_okf=False, db_path="okfgraph.db",
                 bundle_root=".", device="cpu", ingest_mode="text"):
        super().__init__()
        self.files = files
        self.seen_hashes = seen_hashes
        self.extract_images = extract_images
        self._running = True

        # Ingestion settings
        self.ingest_into_okf = ingest_into_okf
        self.db_path = db_path
        self.bundle_root = bundle_root
        self.device = device
        self.ingest_mode = ingest_mode
        self.router = None  # lazy-init on first ingest

    def _compute_hash(self, filepath: Path) -> str:
        """Compute SHA-256 hash to identify identical files."""
        h = hashlib.sha256()
        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    def _ensure_router(self):
        """Lazily create the OKFRouter (expensive — only once per batch)."""
        if self.router is None and OKFRouter is not None:
            self.log_signal.emit(f"⚙️  Loading okfgraph embedding model (device={self.device})...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.router = OKFRouter(
                    db_path=self.db_path,
                    bundle_root=self.bundle_root,
                    embedding_dim=512,
                    device=self.device,
                )
            self.log_signal.emit("✅  Embedding model ready.")

    def run(self):
        success = fail = skipped = ingest_ok = ingest_fail = 0
        total = len(self.files)
        try:
            for idx, (file_path, out_dir) in enumerate(self.files, 1):
                if not self._running:
                    break
                self.progress_signal.emit(idx, total)
                result = self._convert_one(file_path, out_dir)
                if result == "success":
                    success += 1
                elif result == "skipped":
                    skipped += 1
                elif result == "ingest-ok":
                    success += 1
                    ingest_ok += 1
                elif result == "ingest-fail":
                    success += 1  # conversion succeeded, ingest failed
                    ingest_fail += 1
                else:
                    fail += 1
        finally:
            # Rebuild the search indexes once for the whole batch (per-file
            # imports deferred it), then checkpoint + close so a separate
            # reader process can open the DB cleanly. Runs even on cancel/error.
            if self.router is not None:
                try:
                    if self.router.reindex(force=False):
                        self.log_signal.emit("🔁  Rebuilt search indexes for the batch.")
                except Exception as e:
                    self.log_signal.emit(f"⚠️  Index rebuild failed: {e}")
                try:
                    self.router.close()
                except Exception:
                    pass
                self.router = None

        self.finished_signal.emit(success, fail, skipped)
        if ingest_ok or ingest_fail:
            self.log_signal.emit(
                f"📊  Ingestion summary: {ingest_ok} imported, {ingest_fail} ingest errors"
            )

    def _convert_one(self, path: Path, out_dir: Path) -> str:
        ext = path.suffix.lower()

        self.log_signal.emit(f"➡️  <b>{path.name}</b>")

        # Hash check for duplicates
        try:
            file_hash = self._compute_hash(path)
            if file_hash in self.seen_hashes:
                self.log_signal.emit(f"⏩  Skipped identical file (already processed): {path.name}")
                return "skipped"
            self.seen_hashes.add(file_hash)
        except Exception as e:
            self.log_signal.emit(f"❌  Hash check failed for {path.name}: {e}")
            return "fail"

        try:
            extract = self.extract_images
            tmpctx = TemporaryDirectory() if extract else None
            img_dir = Path(tmpctx.name) if tmpctx else None
            try:
                if ext == ".pdf":
                    if PdfDocument is None:
                        raise RuntimeError("pdf_oxide not installed. Run: pip install pdf_oxide")
                    # The context manager ensures native PDF handles are freed.
                    # include_images stays off unless extracting; embed_images is
                    # always False so no base64 is inlined.
                    with PdfDocument(str(path)) as doc:
                        md = doc.to_markdown_all(
                            preserve_layout=False,
                            detect_headings=True,
                            include_images=extract,
                            embed_images=False,
                            image_output_dir=str(img_dir) if extract else None,
                        )

                elif ext in SUPPORTED_EXTS:
                    if OfficeDocument is None:
                        raise RuntimeError("office_oxide not installed. Run: pip install office-oxide")
                    # office_oxide must be closed explicitly; context manager does that
                    # Note: office_oxide.to_markdown() has no image extraction support
                    with OfficeDocument.open(str(path)) as doc:
                        md = doc.to_markdown()

                else:
                    self.log_signal.emit(f"⚠️  Skipped unsupported type: {path.name}")
                    return "fail"

                # Versioning logic to prevent overwrites (decide the stem first so
                # asset ids are scoped to the final concept id).
                base_name = path.stem
                md_path = out_dir / f"{base_name}.md"
                counter = 1
                while md_path.exists():
                    md_path = out_dir / f"{base_name}_v{counter}.md"
                    counter += 1

                # Stage images into _assets/<id>.<ext> and rewrite to okf-asset://
                img_count = 0
                if extract:
                    md, img_count = _stage_images_as_okf_assets(
                        md, img_dir, path, out_dir, md_path.stem
                    )

                md_path.write_text(md, encoding="utf-8")
                # Use .as_uri() for correct file:// links on all platforms
                extra = (
                    f" — {img_count} image(s) → {ASSET_STORE_DIRNAME}/"
                    if img_count else ""
                )
                self.log_signal.emit(
                    f"✅  <a href='{md_path.absolute().as_uri()}'>{md_path.name}</a>{extra}"
                )

                # ── Optional: import into okfgraph ─────────────────────────
                if self.ingest_into_okf:
                    if OKFRouter is None:
                        self.log_signal.emit(
                            "⚠️  okfgraph not installed — skipping ingestion. "
                            "Run: pip install -e ."
                        )
                        return "success"
                    self._ensure_router()
                    try:
                        # Defer index rebuilds: importing many files one-by-one
                        # would otherwise rebuild the whole vector/FTS index per
                        # file. We rebuild once for the batch in run() instead.
                        cid = self.router.import_from_okf(
                            md_path, mode=self.ingest_mode, rebuild_indexes=False
                        )
                        imgs = self.router.list_images(cid)
                        img_suffix = f" ({len(imgs)} image(s) indexed)" if imgs else ""
                        self.log_signal.emit(
                            f"📈  Imported into okfgraph as concept '{cid}'{img_suffix}"
                        )
                        return "ingest-ok"
                    except Exception as e:
                        self.log_signal.emit(f"⚠️  Ingestion failed for {md_path.name}: {e}")
                        return "ingest-fail"

                return "success"
            finally:
                if tmpctx is not None:
                    tmpctx.cleanup()

        except Exception as e:
            self.log_signal.emit(f"❌  {path.name}: {e}")
            return "fail"

    def stop(self):
        """Gracefully request the thread to finish."""
        self._running = False
        self.wait(2000)


class DropArea(QFrame):
    """Drop zone that also allows click-to-browse. Styles reset correctly."""
    files_dropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(150)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)

        self._idle_style = """
            DropArea {
                background-color: #ecf0f1;
                border: 3px dashed #bdc3c7;
                border-radius: 15px;
            }
            QLabel { color: #7f8c8d; font-size: 15px; font-weight: bold; }
        """
        self._active_style = """
            DropArea {
                background-color: #eafaf1;
                border: 3px dashed #2ecc71;
                border-radius: 15px;
            }
            QLabel { color: #27ae60; font-size: 15px; font-weight: bold; }
        """
        self.setStyleSheet(self._idle_style)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        self.label = QLabel("📂  Drop files here or click to browse\n"
                            "PDF, DOCX, XLSX, PPTX, DOC, XLS, PPT")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            self.setStyleSheet(self._active_style)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet(self._idle_style)   # always revert

    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet(self._idle_style)   # revert before checking
        # Accept both files and directories
        paths = [
            u.toLocalFile() for u in event.mimeData().urls()
            if u.isLocalFile()
        ]
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()

    def mousePressEvent(self, event):
        """Click to open a file dialog."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Documents", "",
            "Documents (*.pdf *.docx *.xlsx *.pptx *.doc *.xls *.ppt);;All Files (*)"
        )
        if files:
            self.files_dropped.emit(files)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Oxide → Markdown Converter")
        self.resize(650, 580)

        self.worker = None
        self.pending_files = []
        self.seen_hashes = set()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        # Drop area
        self.drop_area = DropArea()
        self.drop_area.files_dropped.connect(self.add_files)
        layout.addWidget(self.drop_area)

        # Options
        self.images_cb = QCheckBox(
            "Extract images as okf-asset:// links (writes an _assets/ folder next to each .md)"
        )
        self.images_cb.setChecked(True)
        self.images_cb.setStyleSheet("color: #7f8c8d;")
        layout.addWidget(self.images_cb)

        # ── okfgraph ingestion settings (collapsible group) ────────────
        self.ingest_box = QGroupBox("📈 Import into okfgraph (optional)")
        ingest_layout = QVBoxLayout(self.ingest_box)

        self.ingest_cb = QCheckBox("Import converted files into okfgraph after conversion")
        self.ingest_cb.setStyleSheet("color: #2980b9; font-weight: bold;")
        ingest_layout.addWidget(self.ingest_cb)

        form = QFormLayout()
        self.db_path_edit = QLineEdit("okfgraph.db")
        self.db_path_edit.setPlaceholderText("Database path (default: okfgraph.db)")
        form.addRow("Database:", self.db_path_edit)

        self.bundle_root_edit = QLineEdit(".")
        self.bundle_root_edit.setPlaceholderText("Bundle root directory (default: .)")
        form.addRow("Bundle root:", self.bundle_root_edit)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda"])
        self.device_combo.setCurrentText("cpu")
        form.addRow("Device:", self.device_combo)

        self.ingest_mode_combo = QComboBox()
        self.ingest_mode_combo.addItems([
            "text  — alt-text / filename fallback (no omni model)",
            "optional  — omni only for images without alt-text",
            "omni  — every image via multimodal model",
        ])
        self.ingest_mode_combo.setCurrentIndex(0)
        form.addRow("Image mode:", self.ingest_mode_combo)

        ingest_layout.addLayout(form)

        # Show availability hint
        if OKFRouter is None:
            hint = QLabel("⚠️  okfgraph not installed — ingestion will be skipped.\n"
                          "Run: pip install -e .")
            hint.setStyleSheet("color: #e74c3c; font-size: 11px;")
            ingest_layout.addWidget(hint)
        else:
            hint = QLabel("✅  okfgraph available")
            hint.setStyleSheet("color: #27ae60; font-size: 11px;")
            ingest_layout.addWidget(hint)

        self.ingest_box.setCheckable(True)
        self.ingest_box.setChecked(False)
        self.ingest_box.toggled.connect(self._toggle_ingest_ui)
        layout.addWidget(self.ingest_box)

        # Status + buttons row
        btn_row = QHBoxLayout()
        self.status_lbl = QLabel("Ready — drop files to begin.")
        self.status_lbl.setStyleSheet("color: #7f8c8d;")
        btn_row.addWidget(self.status_lbl)

        btn_row.addStretch()

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_all)
        btn_row.addWidget(self.clear_btn)

        self.cancel_btn = QPushButton("⏹ Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_conversion)
        btn_row.addWidget(self.cancel_btn)

        self.convert_btn = QPushButton("▶  Convert")
        self.convert_btn.setEnabled(False)
        self.convert_btn.clicked.connect(self.start_conversion)
        btn_row.addWidget(self.convert_btn)
        layout.addLayout(btn_row)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        # Log output
        self.log_area = QTextBrowser()
        self.log_area.setOpenExternalLinks(True)
        self.log_area.setStyleSheet(
            "font-family: 'SF Mono', Consolas, monospace; "
            "background: #1e1e1e; color: #d4d4d4; "
            "border: 1px solid #3c3c3c; border-radius: 5px; padding: 8px;"
        )
        layout.addWidget(self.log_area, stretch=1)

    def _toggle_ingest_ui(self, checked: bool):
        """Show/hide all widgets inside the ingestion group box."""
        for child in self.ingest_box.findChildren(QWidget):
            child.setVisible(checked)

    def _get_ingest_mode(self) -> str:
        """Extract the mode keyword from the combo box display text."""
        text = self.ingest_mode_combo.currentText().split("—")[0].split("  ")[0].strip()
        return text

    def add_files(self, paths: list):
        """Queue supported file types, traversing directories."""
        added_count = 0
        for p in paths:
            path = Path(p)
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
                item = (path, path.parent)
                if item not in self.pending_files:
                    self.pending_files.append(item)
                    added_count += 1
            elif path.is_dir():
                out_dir = path # output goes to the dropped folder's root
                for f in path.rglob('*'):
                    if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS:
                        item = (f, out_dir)
                        if item not in self.pending_files:
                            self.pending_files.append(item)
                            added_count += 1

        if self.pending_files:
            self.convert_btn.setEnabled(True)
            self.status_lbl.setText(f"{len(self.pending_files)} file(s) queued.")
            if added_count > 0:
                self.log(f"📥  Queued {added_count} new file(s).")

    def clear_all(self):
        """Reset the UI and file list (disabled during conversion)."""
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Busy", "A conversion is in progress. Please wait for it to finish.")
            return
        self.pending_files.clear()
        self.seen_hashes.clear()
        self.log_area.clear()
        self.progress.setValue(0)
        self.convert_btn.setEnabled(False)
        self.status_lbl.setText("Ready — drop files to begin.")

    def log(self, msg: str):
        self.log_area.append(msg)

    def cancel_conversion(self):
        if self.worker and self.worker.isRunning():
            self.status_lbl.setText("Cancelling...")
            self.cancel_btn.setEnabled(False)
            self.worker.stop()
            self.log("⏹  Conversion cancelled by user.")

    def start_conversion(self):
        if not self.pending_files:
            return
        self.convert_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.drop_area.setEnabled(False)
        self.images_cb.setEnabled(False)
        self.ingest_cb.setEnabled(False)

        self.worker = ConversionWorker(
            self.pending_files.copy(),
            self.seen_hashes,
            extract_images=self.images_cb.isChecked(),
            ingest_into_okf=self.ingest_cb.isChecked(),
            db_path=self.db_path_edit.text() or "okfgraph.db",
            bundle_root=self.bundle_root_edit.text() or ".",
            device=self.device_combo.currentText(),
            ingest_mode=self._get_ingest_mode(),
        )
        self.worker.log_signal.connect(self.log)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def update_progress(self, current: int, total: int):
        if total > 0:
            self.progress.setValue(int(current / total * 100))
            self.status_lbl.setText(f"Converting {current} / {total}...")

    def on_finished(self, success: int, fail: int, skipped: int):
        self.convert_btn.setEnabled(bool(self.pending_files))
        self.clear_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.drop_area.setEnabled(True)
        self.images_cb.setEnabled(True)
        self.ingest_cb.setEnabled(True)
        self.progress.setValue(100 if fail == 0 else 0)

        summary_msg = f"{success} succeeded\n{fail} failed\n{skipped} skipped (duplicates)."

        if fail == 0:
            self.status_lbl.setText(f"Done! {success} converted, {skipped} skipped.")
            QMessageBox.information(self, "Done", f"Conversion complete:\n{summary_msg}")
        else:
            self.status_lbl.setText(f"Done: {success} succeeded, {fail} failed, {skipped} skipped.")
            QMessageBox.warning(self, "Done",
                f"Conversion finished with errors:\n{summary_msg}\n\nCheck the log for details.")
        self.worker = None

    def closeEvent(self, event):
        """Stop the worker thread before the window closes."""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")   # modern look
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
