import argparse
import logging

import torch

from bret.data_loaders import (
    GenericDataLoader,
    get_text_dataloader,
    get_training_dataloader,
)
from bret.models import model_factory
from bret.training import BayesianDPRTrainer
from bret.utils import get_checkpoint_file_name, get_query_file, get_root_dir

logger = logging.getLogger(__name__)


def preprocess_key(old_key):
    if "embeddings" in old_key:
        return old_key
    if "norm" in old_key.lower():
        return old_key
    if "pooler" in old_key:
        return old_key
    if old_key.endswith(".weight"):
        return old_key.replace(".weight", ".weight_mean")
    if old_key.endswith(".bias"):
        return old_key.replace(".bias", ".bias_mean")
    return old_key


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s: %(message)s", datefmt="%Y/%m/%d %H:%M:%S", level=logging.INFO
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", choices=["msmarco"])
    parser.add_argument("--training_data_file", default="data/msmarco-train.jsonl")
    parser.add_argument("--model_name", default="bert-base")
    parser.add_argument("--method", default="bret", choices=["dpr", "bret"])
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--encoder_ckpt", default=None)  # If provided, training is resumed from checkpoint.
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--min_lr", type=float, default=5e-8)
    parser.add_argument("--warmup_rate", type=float, default=0.1)
    parser.add_argument("--max_qry_len", type=int, default=32)
    parser.add_argument("--max_psg_len", type=int, default=256)
    parser.add_argument("--output_dir", default="output/trained_encoders")
    args = parser.parse_args()
    logger.info(args.__dict__)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info("Using device: %s", device)
    tokenizer, model = model_factory(args.model_name, args.method, device)
    if args.encoder_ckpt is not None:
        logger.info("Loading pre-trained encoder weights from checkpoint: %s", args.encoder_ckpt)
        sd = torch.load(args.encoder_ckpt, map_location=device)
        sdnew = {}
        for old_key, v in sd.items():
            k = preprocess_key(old_key)
            sdnew[k] = v
        model.load_state_dict(sdnew, strict=False)
    model.train()

    train_dl = get_training_dataloader(args.training_data_file, batch_size=args.batch_size, shuffle=True)
    query_file = get_query_file(args.dataset_id, split="val")
    val_query_dl = get_text_dataloader(query_file, batch_size=1, shuffle=False)
    corpus_file = "data/msmarco-corpus-val.jsonl"
    val_corpus_dl = get_text_dataloader(corpus_file, batch_size=args.batch_size, shuffle=False)
    dataset_dir = get_root_dir(args.dataset_id)
    qrels = GenericDataLoader(dataset_dir, split="val").load_qrels()
    ckpt_file_name = get_checkpoint_file_name(args.output_dir, args.model_name, method=args.method)
    trainer = BayesianDPRTrainer(tokenizer, model, train_dl, val_query_dl, val_corpus_dl, qrels, device)
    trainer.train(
        num_epochs=args.num_epochs,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_rate=args.warmup_rate,
        ckpt_file_name=ckpt_file_name,
        num_samples=args.num_samples,
        max_qry_len=args.max_qry_len,
        max_psg_len=args.max_psg_len,
    )
    logger.info("Training finished after %d epochs.", args.num_epochs)


if __name__ == "__main__":
    main()
