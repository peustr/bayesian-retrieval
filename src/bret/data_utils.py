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
