import argparse
import json
import logging
import time

import numpy as np
import torch
from pytrec_eval import RelevanceEvaluator

from bret import BayesianBERTRetriever, BERTRetriever
from bret.data_loaders import GenericDataLoader, make_query_data_loader
from bret.file_utils import get_embedding_file_name, get_results_file_name
from bret.indexing import FaissIndex
from bret.model_utils import get_hf_model_id

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
    parser.add_argument("--model_name", default="bert-tiny")
    parser.add_argument("--encoder_ckpt", default="output/trained_encoders/bert-tiny.pt")
    parser.add_argument("--method", default=None, choices=["vi"])
    parser.add_argument("--max_qry_len", type=int, default=32)
    parser.add_argument("--k", type=int, default=20)  # k as in: nDCG@k.
    parser.add_argument("--embeddings_dir", default="output/embeddings")
    parser.add_argument("--output_dir", default="output/results")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info("Using device: %s", device)
    if args.method == "vi":
        logger.info("Instantiating a Bayesian BERT retriever trained on MS-MARCO with variational inference.")
        tokenizer, model = BayesianBERTRetriever.build(get_hf_model_id(args.model_name), device=device)
    else:
        logger.info("Instantiating a BERT retriever trained on MS-MARCO.")
        tokenizer, model = BERTRetriever.build(get_hf_model_id(args.model_name), device=device)
    logger.info("Loading pre-trained weights from checkpoint: %s", args.encoder_ckpt)
    model.load_state_dict(torch.load(args.encoder_ckpt))
    model.eval()

    logger.info("Indexing corpus encoded from: %s", args.corpus_file)
    t_start = time.time()
    psg_reps = torch.load(get_embedding_file_name(args.embeddings_dir, args.encoder_ckpt, args.corpus_file))
    index = FaissIndex(model.backbone.config.hidden_size)
    index.add(psg_reps)
    t_end = time.time()
    logger.info("Indexing finished in %.2f minutes.", (t_end - t_start) / 60)

    logger.info("Generating run. Searching corpus with queries from: %s", args.query_file)
    t_start = time.time()
    query_dl = make_query_data_loader(
        tokenizer,
        args.query_file,
        max_qry_len=args.max_qry_len,
        batch_size=1,
        shuffle=False,
    )
    run = {}
    for qry_id, qry in query_dl:
        qry = qry.to(device)
        qry_reps, _ = model(qry, None)
        scores, indices = index.search(qry_reps, args.k)
        qid = str(qry_id[0])
        run[qid] = {}
        for score, psg_id in zip(scores[0], indices[0]):
            run[qid][str(psg_id)] = float(score)
    t_end = time.time()
    logger.info("Run generated in %.2f minutes.", (t_end - t_start) / 60)

    logger.info("Evaluating run...")
    _, _, qrels = GenericDataLoader(args.msmarco_dir, split=args.split).load()
    evaluator = RelevanceEvaluator(qrels, {"ndcg", "map", "recip_rank"})
    results = evaluator.evaluate(run)
    map_at_k = []
    mrr_at_k = []
    ndcg_at_k = []
    for qid, metrics in results.items():
        map_at_k.append(metrics["map"])
        mrr_at_k.append(metrics["recip_rank"])
        ndcg_at_k.append(metrics["ndcg"])
    results_agg = {
        "MAP@{}".format(args.k): float(np.mean(map_at_k)),
        "MRR@{}".format(args.k): float(np.mean(mrr_at_k)),
        "nDCG@{}".format(args.k): float(np.mean(ndcg_at_k)),
    }
    logger.info(results_agg)
    results_file_name = get_results_file_name(args.output_dir, args.encoder_ckpt, args.corpus_file, args.k)
    with open(results_file_name, "w") as fp:
        fp.write(json.dumps(results_agg))
    logger.info("Results stored in: %s", results_file_name)


if __name__ == "__main__":
    main()
