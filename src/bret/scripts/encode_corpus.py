import argparse
import logging
import time

import torch

from bret.data_loaders import make_corpus_data_loader
from bret.data_utils import get_corpus_file
from bret.encoding import encode_corpus
from bret.file_utils import get_embedding_file_name
from bret.models import model_factory

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s: %(message)s", datefmt="%Y/%m/%d %H:%M:%S", level=logging.INFO
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", choices=["msmarco", "nq"])
    parser.add_argument("--model_name", default="bert-base")
    parser.add_argument("--encoder_ckpt", default="output/trained_encoders/bert-base.pt")
    parser.add_argument("--method", default=None, choices=["vi"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=30)  # Only for variational inference.
    parser.add_argument("--max_psg_len", type=int, default=256)
    parser.add_argument("--output_dir", default="output/embeddings")
    args = parser.parse_args()
    logger.info(args.__dict__)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info("Using device: %s", device)
    tokenizer, model = model_factory(args.model_name, args.method, device)
    logger.info("Loading pre-trained encoder weights from checkpoint: %s", args.encoder_ckpt)
    model.load_state_dict(torch.load(args.encoder_ckpt))
    model.eval()

    corpus_file = get_corpus_file(args.dataset_id)
    logger.info(f"Encoding the corpus in: %s", corpus_file)
    t_start = time.time()
    corpus_dl = make_corpus_data_loader(
        tokenizer,
        corpus_file,
        max_psg_len=args.max_psg_len,
        batch_size=args.batch_size,
        shuffle=False,
    )
    psg_embs = encode_corpus(corpus_dl, model, device, args.num_samples)
    torch.save(psg_embs, get_embedding_file_name(args.output_dir, args.encoder_ckpt, corpus_file))
    t_end = time.time()
    logger.info("Encoding the corpus finished in %.2f minutes.", (t_end - t_start) / 60)


if __name__ == "__main__":
    main()
