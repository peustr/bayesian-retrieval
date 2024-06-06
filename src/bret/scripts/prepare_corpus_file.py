import argparse
import json

from tqdm.autonotebook import tqdm

from bret.data_loaders import GenericDataLoader
from bret.data_utils import get_corpus_file, get_root_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", choices=["msmarco", "nq"])
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    args = parser.parse_args()
    dataset_dir = get_root_dir(args.dataset_id)
    corpus_file = get_corpus_file(args.dataset_id)
    corpus = GenericDataLoader(dataset_dir, split=args.split).load_corpus()
    with open(corpus_file, "wt", encoding="utf8") as f_out:
        for k, v in tqdm(corpus.items(), total=len(corpus.keys())):
            json.dump({"id": k, "text": v["text"]}, f_out)
            f_out.write("\n")


if __name__ == "__main__":
    main()
