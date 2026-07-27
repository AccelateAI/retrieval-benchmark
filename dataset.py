from beir import util
from beir.datasets.data_loader import GenericDataLoader
import json

name = "scifact"
path = util.download_and_unzip(
    f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip",
    "beir_data",
)
corpus, queries, qrels = GenericDataLoader(path).load(split="test")

with open("data/corpus.jsonl", "w") as f:
    for did, doc in corpus.items():
        text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
        f.write(json.dumps({"id": did, "text": text}) + "\n")

with open("data/queries.jsonl", "w") as f:
    for qid, q in queries.items():
        f.write(json.dumps({"id": qid, "text": q}) + "\n")

with open("data/qrels.tsv", "w") as f:
    for qid, docs in qrels.items():
        for did, rel in docs.items():
            f.write(f"{qid}\t{did}\t{rel}\n")