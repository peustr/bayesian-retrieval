import argparse
import logging
import os
import time

import torch

from bret.data_loaders import QueryDataLoader
from bret.encoding import encode_queries
from bret.models import model_factory
from bret.utils import get_embedding_file_name, get_query_file

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s: %(message)s", datefmt="%Y/%m/%d %H:%M:%S", level=logging.INFO
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", choices=["msmarco", "nq"])
    parser.add_argument("--split", default="dev", choices=["train-queries", "dev", "test", "val"])
    parser.add_argument("--model_name", default="bert-base")
    parser.add_argument("--encoder_ckpt", default="output/trained_encoders/bert-base.pt")
    parser.add_argument("--method", default="dpr", choices=["dpr", "bret"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=30)  # Only for variational inference.
    parser.add_argument("--max_qry_len", type=int, default=32)
    parser.add_argument("--output_dir", default="output/embeddings")
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
    logger.info("Encoding the queries in: %s", query_file)
    t_start = time.time()
    query_dl = QueryDataLoader(
        tokenizer,
        query_file,
        max_qry_len=args.max_qry_len,
        batch_size=1,
        shuffle=False,
    )
    qry_embs = encode_queries(query_dl, model, device, args.method, args.num_samples)
    torch.save(qry_embs, get_embedding_file_name(args.output_dir, args.encoder_ckpt, query_file))
    t_end = time.time()
    logger.info("Encoding the queries finished in %.2f minutes.", (t_end - t_start) / 60)


if __name__ == "__main__":
    main()
