#!/usr/bin/env python3
"""Create a Python 0.15 / Ladybug 0.22.1 v10 DB fixture for the lbug spike."""
from __future__ import annotations

import argparse
import importlib.util
import logging
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.WARNING)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--dim", type=int, default=3)
    args = ap.parse_args()

    import ladybug as lb

    # Load only schema.py: importing okfgraph.components pulls unrelated deps.
    schema_path = Path(__file__).resolve().parents[2] / "okfgraph" / "components" / "schema.py"
    spec = importlib.util.spec_from_file_location("oracle_schema", schema_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    SchemaManager = mod.SchemaManager

    path = Path(args.db)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = lb.Database(str(path))
    conn = lb.Connection(db)
    schema = SchemaManager(conn, args.dim, nullcontext())
    schema._ensure_schema()
    schema._run_schema_migrations()
    conn.execute(
        "CREATE NODE TABLE IF NOT EXISTS MetaText (key STRING PRIMARY KEY, value STRING)"
    )

    ts = datetime(2024, 1, 2, 3, 4, 5)
    emb = [1.0, 0.0, 0.0]
    conn.execute(
        """
        CREATE (:Concept {
            id: 'alpha', type: 'note', title: 'Alpha', description: 'first concept',
            resource: 'kb/alpha.md', tags: ['parity', 'minimal'],
            timestamp: $ts, body: 'Alpha links to beta', embedding: $emb,
            extra: MAP($extra_keys, $extra_values),
            created_at: $ts, updated_at: $ts,
            effective_at: $ts, content_sig: 'oracle-sig'
        })
        """,
        {"ts": ts, "emb": emb, "extra_keys": ["origin"], "extra_values": ["oracle"]},
    )
    conn.execute(
        """
        CREATE (:Concept {
            id: 'beta', type: 'note', title: 'Beta', description: 'second concept',
            resource: 'kb/beta.md', tags: ['parity'],
            timestamp: $ts, body: 'Beta body', embedding: $emb,
            extra: MAP($extra_keys, $extra_values),
            created_at: $ts, updated_at: $ts,
            effective_at: $ts, content_sig: 'oracle-sig-2'
        })
        """,
        {"ts": ts, "emb": emb, "extra_keys": ["origin"], "extra_values": ["oracle"]},
    )
    conn.execute("CREATE (:Directory {id: 'kb'})")
    conn.execute("MATCH (a:Concept {id: 'alpha'}), (b:Concept {id: 'beta'}) CREATE (a)-[:LINKS_TO {kind: 'mention'}]->(b)")
    conn.execute("MATCH (d:Directory {id: 'kb'}), (c:Concept {id: 'alpha'}) CREATE (d)-[:CONTAINS]->(c)")
    conn.execute(
        """
        CREATE (:Chunk {
            id: 'alpha:0', parent_doc_id: 'alpha', chunk_index: 0,
            chunk_text: 'Alpha links to beta', block_type: 'paragraph',
            start_offset: 0, end_offset: 19, embedding: $emb
        })
        """,
        {"emb": emb},
    )
    conn.execute(
        """
        CREATE (:ImageAsset {
            id: 'img-1', file_name: 'pixel.png', mime_type: 'image/png',
            alt_text: 'a pixel', caption: '', embed_route: 'caption',
            content_hash: 'deadbeef', data: $blob, embedding: $emb
        })
        """,
        {"blob": b"\x89PNG\r\n\x1a\n", "emb": emb},
    )
    conn.execute("CREATE (:FileHash {path: 'kb/alpha.md', hash: 'hash-alpha'})")
    schema._set_meta("schema_version", 10)
    schema._set_meta("embedding_precision", 1)
    schema._set_meta("embedding_dim", args.dim)
    for key, value in {
        "embedding_model": "google/embeddinggemma-2",
        "embedding_space": "oracle-space",
        "embedding_image_model": "google/embeddinggemma-2",
        "embedding_image_space": "oracle-space",
        "embedding_image_text_space": "oracle-space",
    }.items():
        conn.execute(
            "MERGE (t:MetaText {key: $k}) SET t.value = $v",
            {"k": key, "v": value},
        )

    try:
        conn.execute(
            "CALL CREATE_VECTOR_INDEX('Concept', 'concept_embedding', 'embedding', "
            "mu := 30, ml := 60, metric := 'cosine', efc := 200)"
        )
        conn.execute(
            "CALL CREATE_FTS_INDEX('Concept', 'concept_fts', "
            "['title', 'description', 'body'])"
        )
        print("created vector + fts indexes")
    except Exception as exc:  # extensions may be unavailable offline
        print(f"index creation skipped: {exc}")

    print(f"wrote {path} with schema version 10")
    conn.close()
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
