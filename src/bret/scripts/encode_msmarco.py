import argparse
import logging
import time

import torch

from bret import BayesianBERTRetriever, BERTRetriever
from bret.data_loaders import make_corpus_data_loader, make_query_data_loader
from bret.file_utils import get_embedding_file_name
from bret.model_utils import get_hf_model_id

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s: %(message)s", datefmt="%Y/%m/%d %H:%M:%S", level=logging.INFO
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_file", default="data/msmarco-dev.jsonl")
    parser.add_argument("--skip_queries", action="store_true")
    parser.add_argument("--corpus_file", default="data/msmarco-corpus.jsonl")
    parser.add_argument("--skip_corpus", action="store_true")
    parser.add_argument("--model_name", default="bert-base")
    parser.add_argument("--encoder_ckpt", default="output/trained_encoders/bert-base.pt")
    parser.add_argument("--method", default=None, choices=["vi"])
    parser.add_argument("--max_qry_len", type=int, default=32)
    parser.add_argument("--max_psg_len", type=int, default=256)
    parser.add_argument("--output_dir", default="output/embeddings")
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

    if args.skip_queries:
        logger.info("Skipping the queries...")
    else:
        logger.info("Encoding the queries...")
        t_start = time.time()
        query_dl = make_query_data_loader(
            tokenizer,
            args.query_file,
            max_qry_len=args.max_qry_len,
            shuffle=False,
        )
        qry_embs = []
        for _, qry in query_dl:
            qry = qry.to(device)
            qry_reps, _ = model(qry, None)
            qry_embs.append(qry_reps.detach().cpu())
        qry_embs = torch.cat(qry_embs, dim=0)
        torch.save(qry_embs, get_embedding_file_name(args.output_dir, args.encoder_ckpt, args.query_file))
        t_end = time.time()
        logger.info("Encoding the queries finished in %.2f minutes.", (t_end - t_start) / 60)

    if args.skip_corpus:
        logger.info("Skipping the corpus...")
    else:
        logger.info("Encoding the corpus...")
        t_start = time.time()
        corpus_dl = make_corpus_data_loader(
            tokenizer,
            args.corpus_file,
            max_psg_len=args.max_psg_len,
            shuffle=False,
        )
        psg_embs = []
        for _, psg in corpus_dl:
            psg = psg.to(device)
            _, psg_reps = model(None, psg)
            psg_embs.append(psg_reps.detach().cpu())
        psg_embs = torch.cat(psg_embs, dim=0)
        torch.save(psg_embs, get_embedding_file_name(args.output_dir, args.encoder_ckpt, args.corpus_file))
        t_end = time.time()
        logger.info("Encoding the corpus finished in %.2f minutes.", (t_end - t_start) / 60)


if __name__ == "__main__":
    main()
