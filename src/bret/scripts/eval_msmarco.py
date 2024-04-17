import argparse
import json
import logging
import time

import numpy as np
import torch

from bret.data_loaders import GenericDataLoader, make_query_data_loader
from bret.evaluation import calculate_metrics, generate_run
from bret.file_utils import get_embedding_file_name, get_results_file_name
from bret.indexing import FaissIndex
from bret.models import model_factory

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s: %(message)s", datefmt="%Y/%m/%d %H:%M:%S", level=logging.INFO
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--msmarco_dir", default="data/msmarco")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--query_file", default="data/msmarco-dev.jsonl")
    parser.add_argument("--corpus_file", default="data/msmarco-corpus.jsonl")
    parser.add_argument("--model_name", default="bert-base")
    parser.add_argument("--encoder_ckpt", default="output/trained_encoders/bert-base.pt")
    parser.add_argument("--method", default=None, choices=["vi"])
    parser.add_argument("--num_samples", type=int, default=100)  # Only for variational inference.
    parser.add_argument("--max_qry_len", type=int, default=32)
    parser.add_argument("--k", type=int, default=20)  # k as in: nDCG@k.
    parser.add_argument("--embeddings_dir", default="output/embeddings")
    parser.add_argument("--output_dir", default="output/results")
    args = parser.parse_args()
    logger.info(args.__dict__)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info("Using device: %s", device)
    tokenizer, model = model_factory(args.model_name, args.method, device)
    logger.info("Loading pre-trained weights from checkpoint: %s", args.encoder_ckpt)
    model.load_state_dict(torch.load(args.encoder_ckpt))
    model.eval()

    psg_reps = torch.load(get_embedding_file_name(args.embeddings_dir, args.encoder_ckpt, args.corpus_file))
    index = FaissIndex.build(psg_reps)

    logger.info("Generating run. Searching corpus with queries from: %s", args.query_file)
    t_start = time.time()
    query_dl = make_query_data_loader(
        tokenizer,
        args.query_file,
        max_qry_len=args.max_qry_len,
        batch_size=1,
        shuffle=False,
    )
    run = generate_run(query_dl, model, index, device, k=args.k, method=args.method, num_samples=args.num_samples)
    t_end = time.time()
    logger.info("Run generated in %.2f minutes.", (t_end - t_start) / 60)

    logger.info("Evaluating run...")
    qrels = GenericDataLoader(args.msmarco_dir, split=args.split).load_qrels()
    results = calculate_metrics(run, qrels, metrics={"ndcg", "map", "recip_rank"}, k=args.k)
    results_file_name = get_results_file_name(args.output_dir, args.encoder_ckpt, args.corpus_file, args.k)
    with open(results_file_name, "w") as fp:
        fp.write(json.dumps(results))
    logger.info("Results stored in: %s", results_file_name)


if __name__ == "__main__":
    main()
