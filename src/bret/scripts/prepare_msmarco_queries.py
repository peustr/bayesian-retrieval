import argparse
import json

from tqdm.autonotebook import tqdm

from bret.data_loaders import GenericDataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--msmarco_dir", default="data/msmarco")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--output_file", default="data/msmarco-dev.jsonl")
    args = parser.parse_args()
    _, queries, _ = GenericDataLoader(args.msmarco_dir, split=args.split).load()
    with open(args.output_file, "wt", encoding="utf8") as f_out:
        for k, v in tqdm(queries.items(), total=len(queries.keys())):
            json.dump({"id": k, "text": v}, f_out)
            f_out.write("\n")


if __name__ == "__main__":
    main()