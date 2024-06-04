import argparse
import json

from tqdm.autonotebook import tqdm

from bret.data_loaders import GenericDataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="data/msmarco")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--output_file", default="data/msmarco-corpus.jsonl")
    args = parser.parse_args()
    corpus = GenericDataLoader(args.dataset_dir, split=args.split).load_corpus()
    with open(args.output_file, "wt", encoding="utf8") as f_out:
        for k, v in tqdm(corpus.items(), total=len(corpus.keys())):
            json.dump({"id": k, "text": v["text"]}, f_out)
            f_out.write("\n")


if __name__ == "__main__":
    main()
