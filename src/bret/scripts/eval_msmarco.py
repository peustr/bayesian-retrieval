import argparse
import json
import logging

import torch

from bret.data_loaders import GenericDataLoader, make_query_data_loader
from bret.evaluation import Evaluator
from bret.file_utils import (
    get_embedding_file_name,
    get_results_file_name,
    get_run_file_name,
)
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
    parser.add_argument("--k", type=int, default=10)  # k as in: nDCG@k.
    parser.add_argument("--embeddings_dir", default="output/embeddings")
    parser.add_argument("--run_dir", default="output/runs")
    parser.add_argument("--output_dir", default="output/results")
    args = parser.parse_args()
    logger.info(args.__dict__)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info("Using device: %s", device)
    tokenizer, model = model_factory(args.model_name, args.method, device)
    logger.info("Loading pre-trained weights from checkpoint: %s", args.encoder_ckpt)
    model.load_state_dict(torch.load(args.encoder_ckpt))
    model.eval()

    logger.info("Searching corpus with queries from: %s", args.query_file)
    query_dl = make_query_data_loader(
        tokenizer,
        args.query_file,
        max_qry_len=args.max_qry_len,
        batch_size=1,
        shuffle=False,
    )
    qrels = GenericDataLoader(args.msmarco_dir, split=args.split).load_qrels()
    index = FaissIndex.build(
        torch.load(get_embedding_file_name(args.embeddings_dir, args.encoder_ckpt, args.corpus_file))
    )
    run_file_name = get_run_file_name(args.run_dir, args.encoder_ckpt, args.query_file)
    evaluator = Evaluator(model, index, device, metrics={"ndcg", "map", "recip_rank"})
    results = evaluator.evaluate_retriever(
        query_dl, qrels, k=args.k, num_samples=args.num_samples, run_file=run_file_name
    )
    results_file_name = get_results_file_name(args.output_dir, args.encoder_ckpt, args.corpus_file, args.k)
    with open(results_file_name, "w") as fp:
        fp.write(json.dumps(results))
    logger.info("Done. Results stored in: %s", results_file_name)


if __name__ == "__main__":
    main()
