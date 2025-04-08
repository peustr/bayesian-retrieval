import gzip
import json

from tqdm import tqdm

DATASET_METADATA = {
    "msmarco": {
        "root_dir": "data/msmarco",
        "corpus_file": "data/msmarco-corpus.jsonl",
        "query_file": "data/msmarco-{}.jsonl",
    },
    "nq": {
        "root_dir": "data/nq",
        "corpus_file": "data/nq-corpus.jsonl",
        "query_file": "data/nq-{}.jsonl",
    },
}


def get_root_dir(dataset_id):
    return DATASET_METADATA[dataset_id]["root_dir"]


def get_corpus_file(dataset_id):
    return DATASET_METADATA[dataset_id]["corpus_file"]


def get_query_file(dataset_id, split):
    return DATASET_METADATA[dataset_id]["query_file"].format(split)


def read_jsonl(file):
    with open(file) as reader:
        jj = []
        for line in reader:
            j = json.loads(line)
            jj.append(j)

        return jj


def write_jsonl(ll, path: str):
    with open(path, "w") as writer:
        for l in tqdm(ll, desc=f"Writing jsonl to {path}"):
            writer.write(f"{json.dumps(l)}\n")


def write_json(d, path, indent=None, zipped=False):
    if zipped:
        with gzip.open(path, "wt", encoding="ascii") as zipfile:
            json.dump(d, zipfile, indent=indent)
    else:
        with open(path, "w") as writer:
            json.dump(d, writer, indent=indent)


def read_json(path, zipped=False):
    if zipped:
        with gzip.open(path, "rt", encoding="ascii") as zipfile:
            return json.load(zipfile)
    else:
        with open(path, "r") as reader:
            return json.load(reader)
