"""Surgical re-embed of a knowledge-base copy under google/embeddinggemma-2.

Reads stored texts (Concept: title+description+body; Chunk: chunk_text),
embeds them with the gemma2 CUDA fp16 engine @512, rewrites the embedding
columns in place, re-pins MODEL_META, and leaves everything else untouched.
Never touches the live file: works on a copy.
"""
import os, shutil, sys, time, gc
os.environ["OKFGRAPH_DEVICE"] = "cuda"
os.environ["OKFGRAPH_NO_SERVER"] = "1"
sys.path.insert(0, r"D:/User/Documents/Python/OKFgraph")

WORK = r"C:/Users/Main/AppData/Local/Temp/okf-p0"
SRC = r"D:/User/Documents/Python/OKFgraph/kb.db"
DST = WORK + r"/kb-gemma.db"
for suffix in ("", ".wal"):
    if os.path.exists(DST + suffix):
        os.remove(DST + suffix)
shutil.copy2(SRC, DST)
if os.path.exists(SRC + ".wal"):
    shutil.copy2(SRC + ".wal", DST + ".wal")

import ladybug as lb
import embroider
from okfgraph.components.embedding import (LazyRustEncoder, MODEL_META_KEY,
                                           resolve_ort_dylib)

_dylib = resolve_ort_dylib(warm_gpu=True)
print("ORT dylib:", _dylib)

# drop the vector indexes only: the vector columns are otherwise locked
for tbl, name in (("Concept", "concept_embedding"), ("Chunk", "chunk_embedding")):
    try:
        c.execute(f"CALL DROP_VECTOR_INDEX('{tbl}', '{name}')")
        print(f"dropped {tbl}.{name}")
    except Exception as exc:
        print(f"drop {tbl}.{name}: {type(exc).__name__}: {exc}")

engine = None  # opened lazily below, after the index drops

h = lb.Database(DST); c = lb.Connection(h)
c.execute("CHECKPOINT")
c.execute("LOAD EXTENSION vector;")
print("vector extension loaded")

# drop the vector indexes only: the vector columns are otherwise locked
for tbl, name in (("Concept", "concept_embedding"), ("Chunk", "chunk_embedding")):
    try:
        c.execute(f"CALL DROP_VECTOR_INDEX('{tbl}', '{name}')")
        print(f"dropped {tbl}.{name}")
    except Exception as exc:
        print(f"drop {tbl}.{name}: {type(exc).__name__}: {exc}")

# open the embed engine AFTER the drops
engine = LazyRustEncoder(
    model_id="google/embeddinggemma-2", truncate_dim=512, device="cuda",
    session_factory=lambda: embroider.JinaV5.open(
        "google/embeddinggemma-2", cache_dir=None, device="cuda",
        truncate_dim=512),
    tokenizer_factory=lambda: embroider.JinaTokenizer.open_files(
        "google/embeddinggemma-2", cache_dir=None))

t0 = time.time()
def update(query, text_key, prop):
    rows = c.execute(query).rows_as_dict().get_all()
    n = 0
    for row in rows:
        text = row[text_key] or ""
        if not text.strip():
            continue
        vec = engine.encode(text, task="Document")
        assert len(vec) == 512, len(vec)
        c.execute(
            f"MATCH (k:{row['_label']} {{id: $id}}) SET k.{prop} = $vec",
            parameters={"id": row["id"], "vec": vec},
        )
        n += 1
    return n

crows = c.execute(
    "MATCH (k:Concept) RETURN k.id AS id, k.title AS t, k.description AS d, "
    "k.body AS b").rows_as_dict().get_all()
n = 0
for r in crows:
    text = f"{r['t'] or ''} {r['d'] or ''} {r['b'] or ''}"
    if not text.strip():
        continue
    vec = engine.encode(text, task="Document")
    assert len(vec) == 512
    c.execute("MATCH (k:Concept {id: $id}) SET k.embedding = $vec",
              parameters={"id": r["id"], "vec": vec})
    n += 1
print(f"concepts re-embedded: {n} ({time.time()-t0:.1f}s)")

chrows = c.execute(
    "MATCH (k:Chunk) RETURN k.id AS id, k.chunk_text AS t").rows_as_dict().get_all()
t1 = time.time()
n = 0
for r in chrows:
    text = r["t"] or ""
    if not text.strip():
        continue
    vec = engine.encode(text, task="Document")
    assert len(vec) == 512
    c.execute("MATCH (k:Chunk {id: $id}) SET k.embedding = $vec",
              parameters={"id": r["id"], "vec": vec})
    n += 1
print(f"chunks re-embedded: {n} ({time.time()-t1:.1f}s)")

# recreate the two vector indexes exactly like the schema lane does
c.execute("CALL CREATE_VECTOR_INDEX('Concept', 'concept_embedding', 'embedding', "
          "mu := 30, ml := 60, metric := 'cosine', efc := 200)")
c.execute("CALL CREATE_VECTOR_INDEX('Chunk', 'chunk_embedding', 'embedding', "
          "mu := 30, ml := 60, metric := 'cosine', efc := 200)")
print("vector indexes recreated")

# re-pin model (precision stays fp16=16)
c.execute(f"MATCH (m:MetaText {{key: '{MODEL_META_KEY}'}}) "
          "SET m.value = 'google/embeddinggemma-2'")
c.execute("CHECKPOINT")
c.close(); del c; h.close(); gc.collect()
print(f"TOTAL {time.time()-t0:.1f}s")
