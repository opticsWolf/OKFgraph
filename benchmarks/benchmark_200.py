"""Performance benchmark: 500 concepts, in-memory DB, synthetic bundle.

Measures:
  - Bundle generation time
  - Single-file import vs batch import (model load excluded from timing)
  - Search latency (hybrid, filtered)
  - Graph traversal scaling
  - Export time

Usage:
    PYTHONPATH=. python benchmarks/benchmark_500.py

Config:
    NUM_CONCEPTS: 50 for quick test, 500 for full benchmark
"""

import json
import statistics
import time
from pathlib import Path

import numpy as np
import yaml

from okfgraph.router import OKFRouter

# ── Config ──────────────────────────────────────────────────────────────────

NUM_CONCEPTS = 200  # 50 for quick test, 200 for full benchmark
DEVICE = "cuda"  # cpu or cuda
DEPTH = 3  # directory nesting depth
SEARCH_ITERATIONS = 5  # warm-up + measurement reps
BATCH_SIZE = 64

# ── Helpers ─────────────────────────────────────────────────────────────────

def _random_text(min_words: int = 240, max_words: int = 600) -> str:
    """Generate synthetic concept body text."""
    words = int(np.random.randint(min_words, max_words))
    vocab = [
        # Core concepts (50)
        "knowledge", "graph", "concept", "node", "edge", "vector", "embedding",
        "search", "retrieve", "index", "query", "semantic", "relationship",
        "traverse", "link", "directory", "bundle", "format", "metadata",
        "property", "attribute", "type", "tag", "label", "description",
        "title", "content", "body", "frontmatter", "yaml", "markdown",
        "database", "storage", "performance", "optimization",
        "model", "inference", "tensor", "dimension", "normalization",
        "pooling", "token", "context", "attention", "transformer",
        "encoder", "decoder", "pretrained", "fine-tuned", "dataset",
        "training", "validation", "testing", "accuracy",
        # ML / NLP (60)
        "precision", "recall", "f1", "score", "ranking", "retrieval",
        "classifier", "regressor", "cluster", "anomaly", "detection",
        "feature", "representation", "latent", "space", "projection",
        "similarity", "distance", "cosine", "euclidean", "manhattan",
        "batch", "epoch", "gradient", "backprop", "loss", "function",
        "optimizer", "adam", "sgd", "momentum", "learning", "rate",
        "dropout", "regularization", "overfit", "underfit", "generalize",
        "cross-entropy", "mse", "mae", "huber", "contrastive", "triplet",
        "augmentation", "synthetic", "sampling", "stratified", "weighted",
        "embedding", "tokenizer", "subword", "bpe", "wordpiece", "sentencepiece",
        "transformer", "bert", "roberta", "gpt", "t5", "distilbert",
        "multihead", "self-attention", "positional", "encoding", "layer",
        "feedforward", "residual", "connection", "normalization", "activation",
        "relu", "gelu", "silu", "swish", "softmax", "logits",
        # Systems / architecture (60)
        "system", "architecture", "design", "pattern", "component",
        "module", "interface", "implementation", "abstraction", "layer",
        "pipeline", "workflow", "orchestration", "deployment", "container",
        "scalability", "throughput", "latency", "benchmark", "profiling",
        "caching", "indexing", "partitioning", "sharding", "replication",
        "consistency", "availability", "durability", "isolation", "atomicity",
        "transaction", "rollback", "commit", "checkpoint", "snapshot",
        "buffer", "queue", "stream", "pipeline", "worker", "thread",
        "process", "memory", "allocation", "garbage", "collection",
        "serialization", "deserialization", "protocol", "buffer", "message",
        "grpc", "rest", "graphql", "websocket", "http", "tcp",
        "firewall", "encryption", "authentication", "authorization", "token",
        "jwt", "oauth", "saml", "certificate", "key", "hash",
        "algorithm", "sha256", "aes", "rsa", "ecdsa", "hmac",
        # Data / types (60)
        "integer", "float", "string", "boolean", "array", "object",
        "dictionary", "map", "set", "list", "tuple", "struct",
        "schema", "table", "column", "row", "record", "field",
        "primary", "foreign", "constraint", "index", "view", "trigger",
        "procedure", "function", "expression", "predicate", "aggregate",
        "projection", "selection", "join", "union", "intersection",
        "difference", "cartesian", "product", "relation", "entity",
        "attribute", "relationship", "cardinality", "hierarchy", "inheritance",
        "polymorphism", "encapsulation", "generic", "template", "abstract",
        "concrete", "virtual", "override", "implement", "extend", "subclass",
        "interface", "protocol", "trait", "mixin", "delegation", "composition",
        # Graph / network (60)
        "adjacency", "incidence", "matrix", "sparse", "dense", "weighted",
        "directed", "undirected", "acyclic", "cycle", "path", "shortest",
        "dijkstra", "bellman", "floyd", "warshall", "topological", "sort",
        "connected", "component", "strongly", "weakly", "bipartite", "tree",
        "forest", "spanning", "minimum", "kruskal", "prim", "art",
        "cut", "vertex", "articulation", "bridge", "flow", "maxflow",
        "mincut", "network", "graph", "neural", "convolutional",
        "recurrent", "lstm", "gru", "attention", "mechanism", "transformer",
        "graph", "neural", "network", "gat", "gcn", "graphsage",
        "node2vec", "deepwalk", "trans", "embedding", "link", "prediction",
        "community", "detection", "modularity", "clustering", "coefficient",
        "degree", "centrality", "betweenness", "closeness", "pagerank",
        "hitting", "set", "influence", "spread", "cascade", "diffusion",
        # Text / language (60)
        "corpus", "vocabulary", "lexicon", "morphology", "syntax", "semantics",
        "pragmatics", "discourse", "narrative", "dialogue", "conversation",
        "utterance", "intent", "entity", "extraction", "classification",
        "sentiment", "emotion", "opinion", "stance", "summarization",
        "translation", "generation", "completion", "filling", "cloze",
        "masking", "denoising", "autoencoder", "variational", "generative",
        "adversarial", "gan", "diffusion", "flow", "normalizing",
        "autoregressive", "non-autoregressive", "decoder", "encoder",
        "seq2seq", "transformer", "attention", "mechanism", "positional",
        "embedding", "word", "phrase", "clause", "sentence", "paragraph",
        "document", "chapter", "section", "subsection", "heading", "footer",
        # Math / stats (60)
        "eigenvalue", "eigenvector", "singular", "decomposition", "svd",
        "principal", "component", "analysis", "pca", "independent", "ica",
        "factor", "analysis", "latent", "dirichlet", "lda", "topic",
        "mixture", "gaussian", "kmeans", "hierarchical", "dbscan",
        "optics", "spectral", "affinity", "propagation", "mean", "shift",
        "probability", "distribution", "gaussian", "normal", "uniform",
        "bernoulli", "binomial", "poisson", "exponential", "gamma",
        "beta", "dirichlet", "multinomial", "categorical", "laplace",
        "student", "t-distribution", "chi-squared", "f-distribution",
        "variance", "covariance", "correlation", "regression", "logistic",
        "linear", "polynomial", "ridge", "lasso", "elastic", "net",
        "kernel", "svm", "naive", "bayes", "decision", "tree",
        "random", "forest", "gradient", "boosting", "xgboost", "lightgbm",
        "categorical", "ordinal", "nominal", "continuous", "discrete",
        "bivariate", "multivariate", "univariate", "marginal", "conditional",
        "posterior", "prior", "likelihood", "evidence", "bayesian", "inference",
        # Computing / infrastructure (60)
        "cpu", "gpu", "tpu", "fpga", "asic", "accelerator", "co-processor",
        "simd", "vectorization", "parallelization", "multithreading", "async",
        "concurrency", "synchronization", "mutex", "semaphore", "lock",
        "deadlock", "livelock", "starvation", "race", "condition", "barrier",
        "pipeline", "staging", "buffering", "prefetching", "caching",
        "l1", "l2", "l3", "cache", "tlb", "page", "fault", "segmentation",
        "virtual", "memory", "paging", "swapping", "thrashing", "compaction",
        "fragmentation", "defragmentation", "garbage", "collection", "mark",
        "sweep", "copying", "generational", "incremental", "concurrent",
        "real-time", "scheduling", "priority", "preemptive", "cooperative",
        "interrupt", "handler", "syscall", "kernel", "user", "space",
        "ring", "privilege", "virtualization", "hypervisor", "container",
        "docker", "kubernetes", "pod", "deployment", "service", "ingress",
        # Domain terms (60)
        "ontology", "taxonomy", "thesaurus", "glossary", "encyclopedia",
        "dictionary", "lexicon", "corpus", "annotation", "labeling",
        "ground", "truth", "supervision", "semi-supervised", "self-supervised",
        "unsupervised", "reinforcement", "imitation", "demonstration", "reward",
        "penalty", "intrinsic", "extrinsic", "sparse", "dense", "shaping",
        "curriculum", "learning", "continual", "lifelong", "incremental",
        "meta-learning", "few-shot", "one-shot", "zero-shot", "prompt",
        "instruction", "chain-of-thought", "retrieval", "augmented", "generation",
        "rag", "vector", "database", "knowledge", "base", "reasoning",
        "planning", "tool", "use", "function", "calling", "agent",
        "orchestration", "multi-agent", "collaboration", "delegation", "coordination",
        "consensus", "voting", "aggregation", "ensemble", "stacking",
        "blending", "bagging", "boosting", "voting", "averaging",
        "interpretability", "explainability", "fairness", "bias", "robustness",
    ]
    return " ".join(str(w) for w in np.random.choice(vocab, size=words, replace=True))


def generate_bundle(bundle_root: Path, num: int = NUM_CONCEPTS, depth: int = DEPTH) -> list[str]:
    """Create a synthetic OKF bundle with `num` concepts across `depth` levels."""
    directories = [""]  # root
    for d in range(1, depth + 1):
        new_dirs = []
        for parent in directories[-1]:
            for i in range(3):
                child = f"{parent}{parent and parent + '/'}dir_{d}_{i}"
                new_dirs.append(child)
        directories.append(new_dirs)

    all_dirs = [d for d in directories if d]  # non-root
    types = ["chapter", "section", "note", "reference", "table"]
    file_paths = []

    for i in range(num):
        # Pick a random directory
        dir_path = str(np.random.choice(all_dirs)) if all_dirs else ""
        concept_id = f"{dir_path}/{dir_path and 'c'}_{i:04d}".lstrip("/")
        concept_type = str(np.random.choice(types))
        title = f"Concept {i}"
        description = f"Auto-generated concept {i} of type {concept_type}"
        tags = [str(t) for t in np.random.choice(["okf", "test", "benchmark", "data", "model"], size=2, replace=False)]

        fm = yaml.dump({
            "type": concept_type,
            "title": title,
            "description": description,
            "tags": tags,
        }, default_flow_style=False)

        rel_path = f"{concept_id}.md" if concept_id else f"c_{i:04d}.md"
        full_path = bundle_root / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(f"---\n{fm}---\n\n{_random_text()}\n", encoding="utf-8")
        file_paths.append(full_path)

    return file_paths


# ── Benchmark suite ─────────────────────────────────────────────────────────

class Benchmark:
    def __init__(self, bundle_root: Path, db_path: str = ":memory:", device: str = DEVICE):
        self.bundle_root = bundle_root
        self.db_path = db_path
        self.device = device
        self.timings: dict[str, list[float]] = {}

    def _record(self, name: str, value: float):
        self.timings.setdefault(name, []).append(value)

    def _summary(self) -> str:
        lines = ["\n" + "=" * 60, "BENCHMARK SUMMARY", "=" * 60]
        for name, vals in sorted(self.timings.items()):
            # Flatten if vals contains lists
            flat_vals = []
            for v in vals:
                if isinstance(v, list):
                    flat_vals.extend(v)
                else:
                    flat_vals.append(v)
            if len(flat_vals) == 1:
                lines.append(f"  {name:<35s} {flat_vals[0]:>8.3f}s")
            else:
                lines.append(
                    f"  {name:<35s} "
                    f"mean={np.mean(flat_vals):.3f}s  "
                    f"median={np.median(flat_vals):.3f}s  "
                    f"min={np.min(flat_vals):.3f}s  "
                    f"max={np.max(flat_vals):.3f}s"
                )
        lines.append("=" * 60)
        return "\n".join(lines)

    # ── 1. Bundle Generation ────────────────────────────────────────────

    def benchmark_bundle_generation(self):
        print("\n" + "-" * 40)
        print(f"Generating {NUM_CONCEPTS} concepts...")
        print("-" * 40)
        t0 = time.perf_counter()
        files = generate_bundle(self.bundle_root)
        elapsed = time.perf_counter() - t0
        self._record("bundle_generation", elapsed)
        print(f"  [OK] {len(files)} files in {elapsed:.3f}s ({elapsed / len(files) * 1000:.1f}ms/file)")
        return files

    # ── 2. Single-File Import ───────────────────────────────────────────

    def benchmark_single_import(self, files: list[Path]):
        print("\n" + "-" * 40)
        print(f"Single-file import ({len(files)} files)...")
        print("-" * 40)
        router = OKFRouter(db_path=self.db_path, bundle_root=self.bundle_root, device=self.device)
        t0 = time.perf_counter()
        for i, f in enumerate(files):
            router.import_from_okf(f)
            if (i + 1) % 100 == 0:
                elapsed = time.perf_counter() - t0
                print(f"    ... {i+1}/{len(files)} ({elapsed:.1f}s)")
        elapsed = time.perf_counter() - t0
        self._record("single_import_total", elapsed)
        self._record("single_import_per_file", elapsed / len(files))
        print(f"  [OK] {len(files)} concepts in {elapsed:.3f}s ({elapsed / len(files) * 1000:.1f}ms/concept)")
        return router

    # ── 3. Batch Import ─────────────────────────────────────────────────

    def benchmark_batch_import(self, files: list[Path]):
        print("\n" + "-" * 40)
        print(f"Batch import ({len(files)} files, batch_size={BATCH_SIZE})...")
        print("-" * 40)
        router = OKFRouter(db_path=self.db_path, bundle_root=self.bundle_root, device=self.device)
        t0 = time.perf_counter()
        ids = router.import_bundle(self.bundle_root, batch_size=BATCH_SIZE)
        elapsed = time.perf_counter() - t0
        self._record("batch_import_total", elapsed)
        self._record("batch_import_per_file", elapsed / len(ids))
        speedup = self.timings["single_import_total"][0] / elapsed if elapsed > 0 else 0
        print(f"  [OK] {len(ids)} concepts in {elapsed:.3f}s ({elapsed / len(ids) * 1000:.1f}ms/concept)")
        print(f"  [OK] Speedup vs single: {speedup:.1f}x")
        return router

    # ── 4. Search Latency ───────────────────────────────────────────────

    def benchmark_search(self, router: OKFRouter):
        print("\n" + "-" * 40)
        print(f"Search latency ({SEARCH_ITERATIONS} reps each)...")
        print("-" * 40)

        queries = [
            "semantic retrieval system",
            "graph traversal pattern",
            "knowledge concept node",
            "vector embedding optimization",
            "database architecture design",
        ]

        for label, method in [
            ("hybrid_search", lambda q: router.search_hybrid(q, limit=10)),
            ("hybrid_search (filtered)", lambda q: router.search_hybrid(q, limit=10, concept_type="note")),
            ("hybrid_search (tags)", lambda q: router.search_hybrid(q, limit=10, tags=["okf"])),
        ]:
            times = []
            for _ in range(SEARCH_ITERATIONS):
                q = str(np.random.choice(queries))
                t0 = time.perf_counter()
                results = method(q)
                elapsed = time.perf_counter() - t0
                times.append(elapsed)
            self._record(label, times)
            print(f"  {label:<35s} mean={np.mean(times):.4f}s  median={np.median(times):.4f}s  ({len(results)} results)")

    # ── 5. Traversal Scaling ────────────────────────────────────────────

    def benchmark_traversal(self, router: OKFRouter):
        print("\n" + "-" * 40)
        print("Graph traversal scaling...")
        print("-" * 40)

        # Get a root directory to traverse from
        result = router.conn.execute("""
            MATCH (d:Directory) RETURN d.id AS id LIMIT 1
        """)
        rows = result.rows_as_dict().get_all()
        if not rows:
            print("  [SKIP] No directories found")
            return

        start_id = rows[0]["id"]
        for depth in [1, 2, 3, 5]:
            times = []
            for _ in range(SEARCH_ITERATIONS):
                t0 = time.perf_counter()
                results = router.traverse(start_id, relationship="CONTAINS", depth=depth)
                elapsed = time.perf_counter() - t0
                times.append(elapsed)
            self._record(f"traverse_depth_{depth}", times)
            print(f"  depth={depth:<2d}  mean={np.mean(times):.4f}s  results={len(results)}")

    # ── 6. Export Time ──────────────────────────────────────────────────

    def benchmark_export(self, router: OKFRouter, export_root: Path):
        print("\n" + "-" * 40)
        print("Export time...")
        print("-" * 40)

        t0 = time.perf_counter()
        ids = router.export_bundle(export_root)
        elapsed = time.perf_counter() - t0
        self._record("export_bundle", elapsed)
        print(f"  [OK] {len(ids)} concepts exported in {elapsed:.3f}s ({elapsed / len(ids) * 1000:.1f}ms/concept)")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    import tempfile

    tmpdir = tempfile.TemporaryDirectory()
    bundle_root = Path(tmpdir.name) / "bundle"
    bundle_root.mkdir()
    export_root = Path(tmpdir.name) / "export"
    export_root.mkdir()

    print("=" * 60)
    print(f"PERFORMANCE BENCHMARK  --  {NUM_CONCEPTS} concepts, in-memory DB, device={DEVICE}")
    print("=" * 60)

    bench = Benchmark(bundle_root, db_path=":memory:", device=DEVICE)

    # 1. Generate bundle
    files = bench.benchmark_bundle_generation()

    # 2. Single-file import
    router = bench.benchmark_single_import(files)

    # 3. Batch import (fresh router)
    router_batch = bench.benchmark_batch_import(files)

    # 4-6: Search, traversal, export (use batch router)
    bench.benchmark_search(router_batch)
    bench.benchmark_traversal(router_batch)
    bench.benchmark_export(router_batch, export_root)

    # Summary
    print(bench._summary())

    # JSON output
    summary_file = Path(tmpdir.name) / "benchmark_results.json"
    summary = {
        "num_concepts": NUM_CONCEPTS,
        "batch_size": BATCH_SIZE,
        "db": "in-memory",
        "timings": {},
    }
    for k, vals in bench.timings.items():
        flat = []
        for v in vals:
            if isinstance(v, list):
                flat.extend([float(x) for x in v])
            else:
                flat.append(float(v))
        summary["timings"][k] = [round(v, 4) for v in flat]
    summary_file.write_text(json.dumps(summary, indent=2))
    print(f"\nFull results -> {summary_file}")

    tmpdir.cleanup()


if __name__ == "__main__":
    main()
