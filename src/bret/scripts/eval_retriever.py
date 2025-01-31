import argparse
import json
import logging
import os

import torch

from bret.data_loaders import GenericDataLoader, get_text_dataloader
from bret.evaluation import Evaluator
from bret.indexing import FaissIndex
from bret.models import model_factory
from bret.utils import (
    get_corpus_file,
    get_embedding_file_name,
    get_query_file,
    get_results_file_name,
    get_root_dir,
    get_run_file_name,
)

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s: %(message)s", datefmt="%Y/%m/%d %H:%M:%S", level=logging.INFO
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", choices=["msmarco", "nq"])
    parser.add_argument("--split", default="dev", choices=["dev", "test", "val"])
    parser.add_argument("--model_name", default="bert-base")
    parser.add_argument("--encoder_ckpt", default="output/trained_encoders/bert-base.pt")
    parser.add_argument("--method", default="dpr", choices=["dpr", "bret"])
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
    if os.path.exists(args.encoder_ckpt) and os.path.isfile(args.encoder_ckpt):
        logger.info("Loading pre-trained encoder weights from checkpoint: %s", args.encoder_ckpt)
        model.load_state_dict(torch.load(args.encoder_ckpt))
    model.eval()

    query_file = get_query_file(args.dataset_id, split=args.split)
    logger.info("Searching corpus with queries from: %s", query_file)
    query_dl = get_text_dataloader(query_file, batch_size=1, shuffle=False)
    dataset_dir = get_root_dir(args.dataset_id)
    corpus_file = get_corpus_file(args.dataset_id)
    qrels = GenericDataLoader(dataset_dir, split=args.split).load_qrels()
    run_file_name = get_run_file_name(args.run_dir, args.encoder_ckpt, query_file, args.k)
    if os.path.exists(run_file_name) and os.path.isfile(run_file_name):
        index = None
    else:
        index = FaissIndex.build(
            torch.load(get_embedding_file_name(args.embeddings_dir, args.encoder_ckpt, corpus_file))
        )
    evaluator = Evaluator(tokenizer, model, args.method, device, index=index, metrics={"ndcg", "recip_rank"})
    results = evaluator.evaluate_retriever(
        query_dl, qrels, k=args.k, num_samples=args.num_samples, max_qry_len=args.max_qry_len, run_file=run_file_name
    )
    results_file_name = get_results_file_name(args.output_dir, args.encoder_ckpt, query_file, args.k)
    with open(results_file_name, "w") as fp:
        fp.write(json.dumps(results))
    logger.info("Done. Results stored in: %s", results_file_name)


if __name__ == "__main__":
    main()
