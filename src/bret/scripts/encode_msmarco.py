import argparse
import logging
import time

import torch

from bret.data_loaders import make_corpus_data_loader, make_query_data_loader
from bret.file_utils import get_embedding_file_name
from bret.models import model_factory

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s: %(message)s", datefmt="%Y/%m/%d %H:%M:%S", level=logging.INFO
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_file", default="data/msmarco-corpus.jsonl")
    parser.add_argument("--model_name", default="bert-base")
    parser.add_argument("--encoder_ckpt", default="output/trained_encoders/bert-base.pt")
    parser.add_argument("--method", default=None, choices=["vi"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=10)  # Only for variational inference.
    parser.add_argument("--max_psg_len", type=int, default=256)
    parser.add_argument("--output_dir", default="output/embeddings")
    args = parser.parse_args()
    logger.info(args.__dict__)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info("Using device: %s", device)
    tokenizer, model = model_factory(args.model_name, args.method, device)
    logger.info("Loading pre-trained weights from checkpoint: %s", args.encoder_ckpt)
    model.load_state_dict(torch.load(args.encoder_ckpt))
    model.eval()

    logger.info("Encoding the MS-MARCO corpus...")
    t_start = time.time()
    corpus_dl = make_corpus_data_loader(
        tokenizer,
        args.corpus_file,
        max_psg_len=args.max_psg_len,
        batch_size=args.batch_size,
        shuffle=False,
    )
    psg_embs = []
    for _, psg in corpus_dl:
        psg = psg.to(device)
        if args.method == "vi":
            _, psg_reps = model(None, psg, args.num_samples)
            psg_reps = psg_reps.mean(1)
        else:
            _, psg_reps = model(None, psg)
        psg_embs.append(psg_reps.detach().cpu())
    psg_embs = torch.cat(psg_embs, dim=0)
    torch.save(psg_embs, get_embedding_file_name(args.output_dir, args.encoder_ckpt, args.corpus_file))
    t_end = time.time()
    logger.info("Encoding the corpus finished in %.2f minutes.", (t_end - t_start) / 60)


if __name__ == "__main__":
    main()
