#!/usr/bin/env python3
"""
retrieval_benchmark.py
======================
Compare four retrieval setups on the same queries and corpus:

    1. BM25            (sparse, keyword)
    2. Dense           (semantic vector search)
    3. Hybrid          (BM25 + Dense fused with Reciprocal Rank Fusion)
    4. Hybrid + Rerank (cross-encoder reranks the hybrid top-N)

It reports Precision@k, Recall@k, nDCG@k, MRR, and mean query latency for
each setup, writes results.csv, and renders results_ndcg.png.

The point of this harness is to produce YOUR OWN real numbers. Run it on a
real dataset (BEIR: scifact / nfcorpus / fiqa, or your own corpus) and the
output becomes the results table in the case study. Nothing here invents
numbers; it measures them.

QUICK CHECK (offline, no downloads, ~1 second):
    python3 retrieval_benchmark.py --selftest

REAL RUN (needs: pip install rank_bm25 sentence-transformers faiss-cpu):
    python3 retrieval_benchmark.py \
        --corpus data/corpus.jsonl \
        --queries data/queries.jsonl \
        --qrels data/qrels.tsv \
        --dense-model sentence-transformers/all-MiniLM-L6-v2 \
        --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
        --rerank-depth 100

Data formats:
    corpus.jsonl  -> one JSON per line: {"id": "d1", "text": "..."}
    queries.jsonl -> one JSON per line: {"id": "q1", "text": "..."}
    qrels.tsv     -> tab-separated lines: query_id <TAB> doc_id <TAB> relevance
                     (relevance is an int; 0 = not relevant, >0 = relevant/graded)
"""

import argparse
import csv
import json
import math
import re
import time
from collections import defaultdict

import numpy as np

# ----------------------------------------------------------------------------
# Tokenisation (simple, deterministic; swap for a real tokenizer if you like)
# ----------------------------------------------------------------------------
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str):
    return _TOKEN.findall(text.lower())


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def _relevant_set(qrels_q):
    return {doc_id for doc_id, rel in qrels_q.items() if rel > 0}


def precision_at_k(ranked, qrels_q, k):
    rel = _relevant_set(qrels_q)
    if k == 0:
        return 0.0
    hits = sum(1 for d in ranked[:k] if d in rel)
    return hits / k


def recall_at_k(ranked, qrels_q, k):
    rel = _relevant_set(qrels_q)
    if not rel:
        return 0.0
    hits = sum(1 for d in ranked[:k] if d in rel)
    return hits / len(rel)


def dcg_at_k(ranked, qrels_q, k):
    dcg = 0.0
    for i, d in enumerate(ranked[:k]):
        gain = qrels_q.get(d, 0)
        if gain > 0:
            dcg += gain / math.log2(i + 2)  # i is 0-indexed -> rank i+1
    return dcg


def ndcg_at_k(ranked, qrels_q, k):
    ideal = sorted(qrels_q.values(), reverse=True)
    idcg = 0.0
    for i, gain in enumerate(ideal[:k]):
        if gain > 0:
            idcg += gain / math.log2(i + 2)
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked, qrels_q, k) / idcg


def mrr(ranked, qrels_q, k=10):
    rel = _relevant_set(qrels_q)
    for i, d in enumerate(ranked[:k]):
        if d in rel:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(rankings, qrels, ks=(1, 5, 10)):
    """rankings: {qid: [docid, ...]}  ->  averaged metric dict."""
    agg = defaultdict(list)
    for qid, ranked in rankings.items():
        q = qrels.get(qid, {})
        if not _relevant_set(q):
            continue  # skip queries with no known relevant docs
        for k in ks:
            agg[f"P@{k}"].append(precision_at_k(ranked, q, k))
            agg[f"R@{k}"].append(recall_at_k(ranked, q, k))
            agg[f"nDCG@{k}"].append(ndcg_at_k(ranked, q, k))
        agg["MRR@10"].append(mrr(ranked, q, 10))
    return {m: float(np.mean(v)) if v else 0.0 for m, v in agg.items()}


# ----------------------------------------------------------------------------
# Retrievers
# ----------------------------------------------------------------------------
class BM25Retriever:
    def __init__(self, corpus_ids, corpus_texts):
        from rank_bm25 import BM25Okapi

        self.ids = corpus_ids
        self.bm25 = BM25Okapi([tokenize(t) for t in corpus_texts])

    def search(self, query, top_k):
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:top_k]
        return [(self.ids[i], float(scores[i])) for i in order]


class DenseRetriever:
    """Real mode uses sentence-transformers. Fake mode uses deterministic
    pseudo-random vectors so the pipeline runs offline for --selftest."""

    def __init__(self, corpus_ids, corpus_texts, model_name, fake=False, dim=64):
        self.ids = corpus_ids
        self.fake = fake
        self.dim = dim
        if fake:
            self.model = None
            self.doc_emb = np.vstack([self._fake_vec(t) for t in corpus_texts])
        else:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            self.doc_emb = self.model.encode(
                corpus_texts, convert_to_numpy=True, normalize_embeddings=True,
                show_progress_bar=False,
            )

    def _fake_vec(self, text):
        seed = abs(hash(text)) % (2**32)
        v = np.random.default_rng(seed).standard_normal(self.dim)
        return v / (np.linalg.norm(v) + 1e-9)

    def _encode_query(self, query):
        if self.fake:
            return self._fake_vec(query)
        return self.model.encode([query], convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False)[0]

    def search(self, query, top_k):
        q = self._encode_query(query)
        sims = self.doc_emb @ q
        order = np.argsort(sims)[::-1][:top_k]
        return [(self.ids[i], float(sims[i])) for i in order]


def reciprocal_rank_fusion(result_lists, k=60, top_k=1000):
    """Fuse several ranked lists of (id, score). RRF ignores raw scores and
    uses rank position, which makes it robust to scale differences."""
    fused = defaultdict(float)
    for results in result_lists:
        for rank, (doc_id, _) in enumerate(results):
            fused[doc_id] += 1.0 / (k + rank + 1)
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return ranked


class Reranker:
    """Cross-encoder reranker. Fake mode reorders by base score (a no-op-ish
    stand-in) so --selftest runs without downloading a model."""

    def __init__(self, model_name, fake=False):
        self.fake = fake
        if fake:
            self.model = None
        else:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(model_name)

    def rerank(self, query, candidates, text_of):
        # candidates: list of (doc_id, base_score)
        if self.fake:
            return sorted(candidates, key=lambda x: x[1], reverse=True)
        pairs = [[query, text_of[d]] for d, _ in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)
        order = np.argsort(scores)[::-1]
        return [(candidates[i][0], float(scores[i])) for i in order]


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_qrels(path):
    qrels = defaultdict(dict)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, did, rel = parts[0], parts[1], parts[2]
            qrels[qid][did] = int(float(rel))
    return qrels


def make_synthetic(n_docs=200, n_queries=40, seed=7):
    """Tiny keyword-driven dataset so retrieval is learnable and the run is
    fast and offline. Each query shares vocabulary with its relevant docs."""
    rng = np.random.default_rng(seed)
    topics = ["cardiac", "renal", "hepatic", "neural", "immune", "vascular",
              "genomic", "microbial", "skeletal", "endocrine"]
    fillers = ["study", "analysis", "model", "patient", "cohort", "trial",
               "method", "result", "evidence", "review", "system", "data"]
    corpus_ids, corpus_texts = [], []
    doc_topic = {}
    for i in range(n_docs):
        t = topics[i % len(topics)]
        words = [t] * 3 + list(rng.choice(fillers, size=12))
        did = f"d{i}"
        corpus_ids.append(did)
        corpus_texts.append(" ".join(words))
        doc_topic[did] = t
    queries, qrels = [], defaultdict(dict)
    for j in range(n_queries):
        t = topics[j % len(topics)]
        qid = f"q{j}"
        queries.append({"id": qid, "text": f"{t} {rng.choice(fillers)}"})
        for did, dt in doc_topic.items():
            if dt == t:
                qrels[qid][did] = 1
    return corpus_ids, corpus_texts, queries, qrels


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------
def timed_search(fn, queries, top_k):
    """Run a search fn over all queries, return {qid: ranked_ids} and mean ms."""
    rankings, elapsed = {}, []
    for q in queries:
        t0 = time.perf_counter()
        results = fn(q["text"], top_k)
        elapsed.append((time.perf_counter() - t0) * 1000.0)
        rankings[q["id"]] = [doc_id for doc_id, _ in results]
    return rankings, float(np.mean(elapsed))


def run(corpus_ids, corpus_texts, queries, qrels, args):
    text_of = dict(zip(corpus_ids, corpus_texts))
    ks = tuple(int(x) for x in args.ks.split(","))
    top_k = max(ks + (10,))
    fake = args.fake_models

    print(f"corpus={len(corpus_ids)}  queries={len(queries)}  fake_models={fake}")

    bm25 = BM25Retriever(corpus_ids, corpus_texts)
    dense = DenseRetriever(corpus_ids, corpus_texts, args.dense_model, fake=fake)
    reranker = Reranker(args.rerank_model, fake=fake)

    def hybrid_search(query, k):
        b = bm25.search(query, args.rerank_depth)
        d = dense.search(query, args.rerank_depth)
        return reciprocal_rank_fusion([b, d], top_k=k)

    def hybrid_rerank_search(query, k):
        fused = hybrid_search(query, args.rerank_depth)
        reranked = reranker.rerank(query, fused, text_of)
        return reranked[:k]

    methods = {
        "BM25": lambda q, k: bm25.search(q, k),
        "Dense": lambda q, k: dense.search(q, k),
        "Hybrid (RRF)": hybrid_search,
        "Hybrid + Rerank": hybrid_rerank_search,
    }

    rows = []
    for name, fn in methods.items():
        rankings, latency_ms = timed_search(fn, queries, top_k)
        metrics = evaluate(rankings, qrels, ks)
        metrics["latency_ms"] = round(latency_ms, 2)
        metrics["method"] = name
        rows.append(metrics)
        head = "  ".join(f"{m}={metrics[m]:.3f}" for m in (f"nDCG@{ks[-1]}", "MRR@10"))
        print(f"  {name:18s} {head}  latency={latency_ms:.2f}ms")

    write_csv(rows, ks, args.out_csv)
    if not args.no_plot:
        try:
            plot_ndcg(rows, ks[-1], args.out_png)
        except Exception as e:  # plotting is optional
            print(f"  (skipped chart: {e})")
    return rows


def write_csv(rows, ks, path):
    cols = ["method"]
    for k in ks:
        cols += [f"P@{k}", f"R@{k}", f"nDCG@{k}"]
    cols += ["MRR@10", "latency_ms"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: (round(r[c], 4) if isinstance(r.get(c), float) else r.get(c, "")) for c in cols})
    print(f"  wrote {path}")


def plot_ndcg(rows, k, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    STONE, ESPRESSO, SIENNA = "#F0EDE8", "#18140D", "#C84B31"
    names = [r["method"] for r in rows]
    vals = [r[f"nDCG@{k}"] for r in rows]
    best = int(np.argmax(vals))
    colors = [SIENNA if i == best else ESPRESSO for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor(STONE)
    ax.set_facecolor(STONE)
    bars = ax.bar(names, vals, color=colors, width=0.6)
    ax.set_ylabel(f"nDCG@{k}", color=ESPRESSO)
    ax.set_title("Retrieval quality by method", color=ESPRESSO, loc="left", fontsize=13)
    ax.tick_params(colors=ESPRESSO)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(ESPRESSO)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", color=ESPRESSO, fontsize=10)
    plt.tight_layout()
    fig.savefig(path, facecolor=STONE)
    print(f"  wrote {path}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Retrieval benchmark: BM25 / Dense / Hybrid / Rerank")
    p.add_argument("--corpus")
    p.add_argument("--queries")
    p.add_argument("--qrels")
    p.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    p.add_argument("--rerank-depth", type=int, default=100,
                   help="how many candidates to fuse/rerank per query")
    p.add_argument("--ks", default="1,5,10", help="comma-separated cutoffs")
    p.add_argument("--out-csv", default="results.csv")
    p.add_argument("--out-png", default="results_ndcg.png")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--fake-models", action="store_true",
                   help="use offline stand-in models (no downloads)")
    p.add_argument("--selftest", action="store_true",
                   help="tiny synthetic dataset + fake models, runs offline")
    args = p.parse_args()

    if args.selftest:
        args.fake_models = True
        cid, ctext, queries, qrels = make_synthetic()
        run(cid, ctext, queries, qrels, args)
        return

    if not (args.corpus and args.queries and args.qrels):
        p.error("provide --corpus, --queries, --qrels (or use --selftest)")

    corpus = load_jsonl(args.corpus)
    cid = [r["id"] for r in corpus]
    ctext = [r["text"] for r in corpus]
    queries = load_jsonl(args.queries)
    qrels = load_qrels(args.qrels)
    run(cid, ctext, queries, qrels, args)


if __name__ == "__main__":
    main()
