"""Before/after probe for the OKFgraph 0.11 findings-store changes.

Runs one scenario and prints a JSON line: 221 files (200 findings + a target
in ev/, 20 decisions in dec/ linking to it), then a no-op import, an edit of
the target, an md ingest of it, single-file imports, a deferred batch import,
search timings, and (0.11 only) the same ops through ``okf serve``.

The encoder is a deterministic stand-in (no model download): it counts every
text embedded, so "texts" columns are exact; timings exclude model work.

    PYTHONPATH=<okfgraph tree> python benchmarks/probe_v011.py LABEL 200

Point PYTHONPATH at a 0.10.2 checkout to get the before numbers.
"""
import hashlib
import json
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

_WORD = re.compile(r"[A-Za-z0-9]+")


def _vec(text, dim):
    acc = np.zeros(dim, dtype="float64")
    for w in _WORD.findall((text or "").lower()) or ["_empty_"]:
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(w.encode()).digest()[:8], "little"))
        acc += rng.standard_normal(dim)
    return (acc / (np.linalg.norm(acc) or 1.0)).astype("float32").tolist()


from okfgraph.components import embedding as E  # noqa: E402

COUNT = [0]


def _dim(self):
    return int(getattr(self, "_truncate_dim", 512) or 512)


def _cb(self, texts, task="Document"):
    COUNT[0] += len(texts)
    return [_vec(t, _dim(self)) for t in texts]


def _ce(self, text, task="Document"):
    COUNT[0] += 1
    return _vec(text, _dim(self))


E.LazyRustEncoder.encode_batch, E.LazyRustEncoder.encode = _cb, _ce
E.LazyRustEncoder.count_tokens = lambda self, text: len((text or "").split())
E.LazyRustEncoder.is_loaded = property(lambda self: True)
from okfgraph import OKFRouter  # noqa: E402

label, n = sys.argv[1], int(sys.argv[2])
WORDS = ['alder', 'birch', 'cedar', 'dogwood', 'elm', 'fir', 'ginkgo', 'hazel', 'ironwood', 'juniper']
root = Path(tempfile.mkdtemp())
kb = root / "kb"


def w(rel, text):
    p = kb / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


w("ev/t.md", "---\ntitle: Target\n---\ntarget body\n")
for i in range(n):
    w(f"ev/f{i:03d}.md", f"---\ntitle: F{i}\n---\nfinding {i} about topic {i % 7}\n")
for i in range(20):
    w(f"dec/d{i:02d}.md", f"---\ntitle: D{i}\n---\nsee [t](../ev/t.md) and [f](../ev/f{i:03d}.md)\n")

r = OKFRouter(db_path=str(root / "kb.db"), bundle_root=str(kb))
out = {"label": label, "files": n + 21}


def timed(fn):
    t = time.perf_counter()
    res = fn()
    return res, round((time.perf_counter() - t) * 1000, 1)


def incoming(target):
    return r.conn.execute(
        "MATCH (a:Concept)-[:LINKS_TO]->(b:Concept {id: $t}) RETURN count(a) AS n",
        {"t": target}).rows_as_dict().get_all()[0]["n"]


COUNT[0] = 0
_, out["first_import_ms"] = timed(lambda: r.import_bundle())
out["first_import_texts"] = COUNT[0]
def median(fn, k=21):
    fn()
    xs = sorted(timed(fn)[1] for _ in range(k))
    return xs[k // 2]
out["search_hybrid_ms"] = median(lambda: r.search("topic 3", limit=10))
out["search_ppr_ms"] = median(lambda: r.search("topic 3", limit=10, rank="ppr"))
out["read_ms"] = median(lambda: r.read("ev/t"))
COUNT[0] = 0
_, out["noop_import_ms"] = timed(lambda: r.import_bundle())
out["noop_import_texts"] = COUNT[0]

w("ev/t.md", "---\ntitle: Target\n---\ntarget body, edited\n")
COUNT[0] = 0
res, out["edit_one_import_ms"] = timed(lambda: r.import_bundle())
out["edit_one_reimported"] = len(res["concept_ids"])
out["edit_one_texts"] = COUNT[0]
out["incoming_links_after_reimport"] = f"{incoming('ev/t')} of 20"

w("ev/t.md", "---\ntitle: Target\n---\ntarget body, ingested\n")
r.ingest(kind="md", md_path=str(kb / "ev/t.md"))
out["incoming_links_after_md_ingest"] = f"{incoming('ev/t')} of 20"
COUNT[0] = 0
res, out["import_after_ingest_ms"] = timed(lambda: r.import_bundle())
out["import_after_ingest_reimported"] = len(res["concept_ids"])
out["import_after_ingest_texts"] = COUNT[0]

# Single-file write cost (what an agent's finding costs).
times = []
for i in range(10):
    p = kb / f"new/n{i}.md"
    w(f"new/n{i}.md", f"---\ntitle: N{i}\n---\nnew finding {i}\n")
    _, ms = timed(lambda p=p: r.import_file(p))
    times.append(ms)
out["import_file_ms_median"] = sorted(times)[len(times) // 2]

if hasattr(r, "import_files"):
    paths = []
    for i in range(10):
        w(f"batch/b{i}.md", f"---\ntitle: B{i}\n---\nbatch finding {WORDS[i]}\n")
        paths.append(kb / f"batch/b{i}.md")
    _, out["import_files_10_defer_ms"] = timed(lambda: r.import_files(paths, rebuild="defer"))
    hits = r.search("hazel", limit=3)  # b7
    out["search_finds_deferred"] = hits[0]["id"] if hits else None

r.close()

# O4: the same ops through okf serve (0.11 only).
try:
    from okfgraph.client import OKFClient
    from okfgraph.serve import serve
    from okfgraph.settings import Settings
except ImportError:
    pass
else:
    ready, stop = threading.Event(), threading.Event()
    sock = root / "okf.sock"
    t = threading.Thread(target=serve, args=(Settings(db_path=str(root / "kb.db"),
                                                      bundle_root=str(kb)), f"unix://{sock}"),
                         kwargs={"ready": ready, "stop": stop}, daemon=True)
    t.start()
    ready.wait(60)
    c = OKFClient(f"unix://{sock}")
    c.search("topic 3", limit=10)
    lat = []
    for _ in range(20):
        _, ms = timed(lambda: c.search("topic 3", limit=10, rank="ppr"))
        lat.append(ms)
    out["serve_search_ppr_ms_median"] = sorted(lat)[10]
    lat = []
    for _ in range(20):
        _, ms = timed(lambda: c.read("ev/t"))
        lat.append(ms)
    out["serve_read_ms_median"] = sorted(lat)[10]
    stop.set()
    t.join(30)

print(json.dumps(out))
