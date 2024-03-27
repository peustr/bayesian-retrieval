import argparse
import gzip
import json

from tqdm.autonotebook import tqdm

from bret.data_loaders import GenericDataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--msmarco_dir", type=str, default="data/msmarco")
    parser.add_argument("--negatives_file", type=str, default="data/msmarco-hard-negatives.jsonl.gz")
    parser.add_argument("--output_file", type=str, default="data/msmarco-train.jsonl")
    args = parser.parse_args()
    corpus, queries, qrels = GenericDataLoader(args.msmarco_dir, split="train").load()
    train_queries = {}
    ce_score_margin = 3
    with gzip.open(args.negatives_file, "rt", encoding="utf8") as f_in:
        for line in tqdm(f_in, total=502939):
            data = json.loads(line)
            # Get the positive passage ids
            pos_pids = [item["pid"] for item in data["pos"]]
            for pid in pos_pids:
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
                    if pid not in neg_pids:
                        neg_pids.add(pid)
            neg_pids = list(neg_pids)
            if len(pos_pids) > 0 and len(neg_pids) > 0:
                train_queries[data["qid"]] = {
                    "query": queries[data["qid"]],
                    "pos": [corpus[pid]["text"] for pid in pos_pids],
                    "neg": [corpus[pid]["text"] for pid in neg_pids],
                }
    with open(args.output_file, "wt", encoding="utf8") as f_out:
        for k, v in tqdm(train_queries.items(), total=len(train_queries.keys())):
            json.dump({"query_id": k, "query": v["query"], "pos": v["pos"], "neg": v["neg"]}, f_out)
            f_out.write("\n")


if __name__ == "__main__":
    main()
