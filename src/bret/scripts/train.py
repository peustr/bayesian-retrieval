import argparse
import logging

import torch

from bret import BERTRetriever
from bret.data_loaders import make_data_loader
from bret.file_utils import get_checkpoint_file_name, get_tokenizer_cache_file_name
from bret.trainers import DPRTrainer

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s: %(message)s", datefmt="%Y/%m/%d %H:%M:%S", level=logging.INFO
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data_file", type=str, default="data/msmarco-train.jsonl")
    parser.add_argument("--model_name", type=str, default="google-bert/bert-base-uncased")
    parser.add_argument("--tokenizer_cache_dir", type=str, default="cache/tokenizer")
    parser.add_argument("--num_train_qry", type=int, default=8)
    parser.add_argument("--num_train_psg", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_qry_len", type=int, default=32)
    parser.add_argument("--max_psg_len", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="output/")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info("Training a BERT retriever for DPR on MS-MARCO.")
    logger.info("Using device: %s", device)
    tokenizer, model = BERTRetriever.build(args.model_name, device=device)
    train_dl = make_data_loader(
        tokenizer,
        args.training_data_file,
        max_qry_len=args.max_qry_len,
        max_psg_len=args.max_psg_len,
        num_train_qry=args.num_train_qry,
        num_train_psg=args.num_train_psg,
        split="train",
        tokenizer_cache_file_name=get_tokenizer_cache_file_name(args.tokenizer_cache_dir, args.model_name),
    )
    DPRTrainer.train(
        model,
        train_dl,
        device,
        lr=args.lr,
        num_epochs=args.num_epochs,
        ckpt_file_name=get_checkpoint_file_name(args.output_dir, args.model_name),
    )
    logger.info("Training finished after %d epochs.", args.num_epochs)


if __name__ == "__main__":
    main()
