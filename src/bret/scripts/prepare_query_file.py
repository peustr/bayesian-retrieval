import argparse
import json

from tqdm.autonotebook import tqdm

from bret.data_loaders import GenericDataLoader
from bret.data_utils import get_query_file, get_root_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", choices=["msmarco", "nq"])
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    args = parser.parse_args()
    dataset_dir = get_root_dir(args.dataset_id)
    query_file = get_query_file(args.dataset_id, split=args.split)
    queries = GenericDataLoader(dataset_dir, split=args.split).load_queries()
    with open(query_file, "wt", encoding="utf8") as f_out:
        for k, v in tqdm(queries.items(), total=len(queries.keys())):
            json.dump({"id": k, "text": v}, f_out)
            f_out.write("\n")


if __name__ == "__main__":
    main()
