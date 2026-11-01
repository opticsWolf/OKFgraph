"""WS2: thought titles are unique per topic and the minted id is visible.

Explicit ``title`` on a thought is kept, and a clash inside the topic is
refused. Without one, the title is the first sentence, with ``(2)``, ``(3)``
suffixes on clashes. A ``[[title]]`` wikilink from a markdown concept resolves
to the thought.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from okfgraph.components.ingest import THOUGHT_TITLE_MAX, _auto_thought_title
from okfgraph.errors import OKFError
from okfgraph.router import OKFRouter

REPO = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 11, 1, 9, 30)


def _router(tmp_dir, name="t.db"):
    return OKFRouter(
        db_path=str(Path(tmp_dir) / name),
        bundle_root=str(tmp_dir),
        embedding_dim=512,
        chunk_size=50,
        chunk_overlap=10,
        device="cpu",
    )


@pytest.fixture
def router():
    tmp = tempfile.mkdtemp()
    r = _router(tmp)
    try:
        yield r
    finally:
        r.close()
        shutil.rmtree(tmp, ignore_errors=True)


# -- pure title rules -------------------------------------------------------

def test_auto_title_is_the_first_sentence():
    text = "Ladybug 0.22.1 fixes the FTS crash. The vector index loop remains."
    assert _auto_thought_title(text, "ladybug", NOW) == \
        "Ladybug 0.22.1 fixes the FTS crash"


def test_auto_title_is_cut_at_a_word_boundary():
    text = "word " * 40
    title = _auto_thought_title(text, "t", NOW)
    assert len(title) <= THOUGHT_TITLE_MAX
    assert title == " ".join(["word"] * (len(title.split())))  # whole words only


def test_auto_title_falls_back_to_topic_and_time_without_a_sentence():
    assert _auto_thought_title("?!  ..", "crash", NOW) == "crash — 2026-11-01 09:30"


# -- stored titles ----------------------------------------------------------

def test_explicit_title_is_stored(router):
    cid = router.ingest("thoughts", thoughts="quokkas live on rottnest",
                        topic="marsupials", title="Quokka habitat")["concept_id"]
    assert router.read(cid)["title"] == "Quokka habitat"


def test_duplicate_explicit_title_in_topic_is_refused(router):
    first = router.ingest("thoughts", thoughts="one", topic="notes",
                          title="Shared name")["concept_id"]
    with pytest.raises(OKFError) as e:
        router.ingest("thoughts", thoughts="two", topic="notes", title="Shared name")
    assert e.value.code == "BAD_VALUE"
    assert e.value.fields["existing_id"] == first
    assert first in e.value.remedy


def test_same_explicit_title_in_another_topic_is_allowed(router):
    router.ingest("thoughts", thoughts="one", topic="alpha", title="Same name")
    cid = router.ingest("thoughts", thoughts="two", topic="beta",
                        title="Same name")["concept_id"]
    assert router.read(cid)["title"] == "Same name"


def test_updating_a_thought_by_id_keeps_its_own_title(router):
    cid = router.ingest("thoughts", thoughts="one", topic="notes",
                        title="Keep me")["concept_id"]
    again = router.ingest("thoughts", thoughts="one, revised", topic="notes",
                          title="Keep me", concept_id=cid)["concept_id"]
    assert again == cid


def test_auto_titles_get_suffixes_on_clash(router):
    text = "Same opening sentence. Then more."
    titles = [router.ingest("thoughts", thoughts=text, topic="loop")["title"]
              for _ in range(3)]
    assert titles == ["Same opening sentence", "Same opening sentence (2)",
                      "Same opening sentence (3)"]


def test_wikilink_to_a_thought_resolves(router):
    router.ingest("thoughts", thoughts="ladybug notes", topic="ladybug",
                  title="Ladybug install notes")
    md = Path(router.bundle_root) / "linker.md"
    md.write_text("---\ntitle: Linker\n---\nSee [[Ladybug install notes]].\n",
                  encoding="utf-8")
    router.ingest("md", md_path=str(md))
    targets = {b["target"] for b in router.list_broken_links()}
    assert "Ladybug install notes" not in targets


def test_cli_prints_the_minted_id_last(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(REPO))
    cmd = [sys.executable, "-m", "okfgraph.cli", "ingest", "--kind", "thoughts",
           "--topic", "cli", "--thoughts", "A sentence for the cli test.",
           "--db-path", str(tmp_path / "c.db"), "--bundle-root", str(tmp_path)]
    out = subprocess.run(cmd, capture_output=True, text=True, env=env,
                         timeout=300, cwd=str(tmp_path))
    assert out.returncode == 0, out.stderr[-800:]
    last = out.stdout.strip().splitlines()[-1]
    assert last.startswith("[OK] ingested thought")
    assert last.rsplit(" - ", 1)[1].startswith("thoughts/cli/")
