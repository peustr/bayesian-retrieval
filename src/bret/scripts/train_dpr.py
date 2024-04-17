import argparse
import logging

import torch

from bret.data_loaders import make_training_data_loader
from bret.file_utils import get_checkpoint_file_name
from bret.models import model_factory
from bret.training import BayesianDPRTrainer, DPRTrainer

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s: %(message)s", datefmt="%Y/%m/%d %H:%M:%S", level=logging.INFO
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data_file", default="data/msmarco-train.jsonl")
    parser.add_argument("--model_name", default="bert-base")
    parser.add_argument("--method", default=None, choices=["vi"])
    parser.add_argument("--encoder_ckpt", default=None)  # If provided, training is resumed from checkpoint.
    parser.add_argument("--num_train_qry", type=int, default=8)
    parser.add_argument("--num_train_psg", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--max_qry_len", type=int, default=32)
    parser.add_argument("--max_psg_len", type=int, default=256)
    parser.add_argument("--output_dir", default="output/trained_encoders")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info("Using device: %s", device)
    tokenizer, model = model_factory(args.model_name, args.method, device)
    if args.encoder_ckpt is not None:
        logger.info("Loading pre-trained weights from checkpoint: %s", args.encoder_ckpt)
        sd = torch.load(args.encoder_ckpt)
        if args.method == "vi":
            sdnew = {}
            for k, v in sd.items():
                if k.endswith(".weight"):
                    sdnew[k.replace(".weight", ".weight_mean")] = v
            model.load_state_dict(sdnew, strict=False)
        else:
            model.load_state_dict(sd)
    model.train()

    train_dl = make_training_data_loader(
        tokenizer,
        args.training_data_file,
        max_qry_len=args.max_qry_len,
        max_psg_len=args.max_psg_len,
        num_train_qry=args.num_train_qry,
        num_train_psg=args.num_train_psg,
        shuffle=True,
    )
    ckpt_file_name = get_checkpoint_file_name(args.output_dir, args.model_name, method=args.method)
    if args.method == "vi":
        BayesianDPRTrainer.train(
            model,
            train_dl,
            device,
            num_epochs=args.num_epochs,
            lr=args.lr,
            gamma=args.gamma,
            ckpt_file_name=ckpt_file_name,
        )
    else:
        DPRTrainer.train(
            model,
            train_dl,
            device,
            num_epochs=args.num_epochs,
            lr=args.lr,
            gamma=args.gamma,
            ckpt_file_name=ckpt_file_name,
        )
    logger.info("Training finished after %d epochs.", args.num_epochs)


if __name__ == "__main__":
    main()
