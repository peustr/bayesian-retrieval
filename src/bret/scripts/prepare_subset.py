import argparse
import gzip
import json
import os
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from bret.data_loaders import GenericDataLoader
from bret.utils import data_utils


def process_negatives():
    corpus, queries, qrels = GenericDataLoader(output_path, split="train").load()
    # read in the negatives file and create the final train file
    train_queries = {}
    ce_score_margin = 3
    with gzip.open(args.negatives, "rt", encoding="utf8") as f_in:
        for line in tqdm(f_in, total=502939):
            data = json.loads(line)
            if data["qid"] not in queries:
                continue
            # Get the positive passage ids
            pos_pids = [item["pid"] for item in data["pos"]]
            for pid in pos_pids:
                if pid not in corpus:
                    continue
                assert qrels[data["qid"]][pid]
            pos_min_ce_score = min([item["ce-score"] for item in data["pos"]])
            ce_score_threshold = pos_min_ce_score - ce_score_margin
            # Get the hard negatives
            neg_pids = set()
            for system_negs in data["neg"].values():
                for item in system_negs:
                    if item["ce-score"] > ce_score_threshold:
                        continue
                    pid = item["pid"]
                    if pid not in corpus:
                        continue
                    if pid not in neg_pids:
                        neg_pids.add(pid)
            neg_pids = list(neg_pids)
            if len(pos_pids) > 0 and len(neg_pids) > 0:
                train_queries[data["qid"]] = {
                    "query": queries[data["qid"]],
                    "pos": [corpus[pid]["text"] for pid in pos_pids],
                    "neg": [corpus[pid]["text"] for pid in neg_pids],
                }

    with open(args.output_train, "wt", encoding="utf8") as f_out:
        for k, v in tqdm(train_queries.items(), total=len(train_queries.keys())):
            json.dump({"query_id": k, "query": v["query"], "pos": v["pos"], "neg": v["neg"]}, f_out)
            f_out.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input dataset", default="data/msmarco")
    parser.add_argument("--output", help="output dataset", default="data/msmarco-small")
    parser.add_argument("--negatives", default="data/msmarco-hard-negatives.jsonl.gz")
    parser.add_argument("--output_train", help="output trainfile", default="data/msmarco-small-train.jsonl")
    parser.add_argument("--size", type=int, help="number of train / val samples")
    parser.add_argument("--corpus_size", type=int, help="number of corpus documents")
    parser.add_argument("--splits", default="dev,test,train,val", help="csv of splits")
    args = parser.parse_args()

    splits = args.splits.split(",")
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "qrels").mkdir(parents=True, exist_ok=True)

    # read queries
    queries = data_utils.read_jsonl(input_path / "queries.jsonl")
    queries_by_id = {_["_id"]: _ for _ in queries}
    qids = set()
    doc_ids_to_grab = set()

    all_qrels = dict()
    # grab qrels
    for split in splits:
        qrels = defaultdict(dict)
        with open(input_path / "qrels" / (split + ".tsv")) as reader:
            for i, line in enumerate(reader):
                if i == 0:
                    continue
                if len(qrels) >= args.size:
                    break
                (query_id, document_id, score) = line.strip().split()
                score = int(score)
                qrels[query_id][document_id] = score
                doc_ids_to_grab.add(document_id)
                qids.add(query_id)

        all_qrels.update(qrels)
        with open(output_path / "qrels" / (split + ".tsv"), "w") as writer:
            for query_id, qrel in qrels.items():
                for doc_id, score in qrel.items():
                    writer.write(f"{query_id}\t{doc_id}\t{score}\n")

    ce_score_margin = 3
    with gzip.open(args.negatives, "rt", encoding="utf8") as f_in:
        for line in tqdm(f_in, total=502939):
            data = json.loads(line)
            if data["qid"] not in qids:
                continue
            # Get the positive passage ids
            pos_pids = [item["pid"] for item in data["pos"]]
            for pid in pos_pids:
                assert all_qrels[data["qid"]][pid]
            pos_min_ce_score = min([item["ce-score"] for item in data["pos"]])
            ce_score_threshold = pos_min_ce_score - ce_score_margin
            # Get the hard negatives
            neg_pids = set()
            for system_negs in data["neg"].values():
                for item in system_negs:
                    if item["ce-score"] > ce_score_threshold:
                        continue
                    pid = item["pid"]
                    if pid not in neg_pids:
                        neg_pids.add(pid)
            neg_pids = list(neg_pids)
            if len(pos_pids) > 0 and len(neg_pids) > 0:
                for pid in pos_pids + neg_pids:
                    doc_ids_to_grab.add(pid)

    # grab documents
    corpus = []
    with open(input_path / "corpus.jsonl") as reader:
        for line in reader:
            doc = json.loads(line)
            if doc["_id"] in doc_ids_to_grab or len(corpus) < args.corpus_size:
                corpus.append(line)

    with open(output_path / "corpus.jsonl", "w") as writer:
        for line in corpus:
            writer.write(line)

    data_utils.write_jsonl([v for (k, v) in queries_by_id.items() if k in qids], output_path / "queries.jsonl")
    process_negatives()
