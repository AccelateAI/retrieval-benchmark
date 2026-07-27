# retrieval-benchmark

Compares four retrieval setups on the same queries and corpus:

1. **BM25** (sparse, keyword)
2. **Dense** (semantic vector search)
3. **Hybrid** (BM25 + Dense fused with Reciprocal Rank Fusion)
4. **Hybrid + Rerank** (cross-encoder reranks the hybrid top-N)

For each setup it reports Precision@k, Recall@k, nDCG@k, MRR, and mean query
latency, then writes `results.csv` and renders `results_ndcg.png`.

## Setup

```bash
./setup.sh
source .venv/bin/activate
```

## Quick check (offline, no downloads, ~1 second)

```bash
python3 retrieval_benchmark.py --selftest
```

## Full run

`dataset.py` downloads the [BEIR](https://github.com/beir-cellar/beir)
`scifact` dataset and converts it into the `id`/`text` JSONL format the
benchmark expects:

```bash
python3 dataset.py
```

```bash
python3 retrieval_benchmark.py \
        --corpus data/corpus.jsonl \
        --queries data/queries.jsonl \
        --qrels data/qrels.tsv \
        --dense-model sentence-transformers/all-MiniLM-L6-v2 \
        --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
        --rerank-depth 100
```

### Data formats

```
corpus.jsonl  -> one JSON per line: {"id": "d1", "text": "..."}
queries.jsonl -> one JSON per line: {"id": "q1", "text": "..."}
qrels.tsv     -> tab-separated lines: query_id <TAB> doc_id <TAB> relevance
                 (relevance is an int; 0 = not relevant, >0 = relevant/graded)
```

## Sample results (BEIR scifact, 300 queries)

| Method            | nDCG@10 | MRR@10 | Latency  |
|-------------------|---------|--------|----------|
| BM25              | 0.652   | 0.618  | 15ms     |
| Dense             | 0.645   | 0.605  | 27ms     |
| Hybrid (RRF)      | 0.684   | 0.650  | 34ms     |
| Hybrid + Rerank   | 0.691   | 0.659  | 1934ms   |
