import argparse
import logging

import torch

from bret.utils import get_corpus_file, get_embedding_file_name

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s: %(message)s", datefmt="%Y/%m/%d %H:%M:%S", level=logging.INFO
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", choices=["msmarco", "nq"])
    parser.add_argument("--encoder_ckpt", default="output/trained_encoders/bert-base.pt")
    parser.add_argument("--num_shards", type=int, default=20)
    parser.add_argument("--embeddings_dir", default="output/embeddings")
    args = parser.parse_args()
    logger.info(args.__dict__)

    corpus_file = get_corpus_file(args.dataset_id)
    psg_embs = []
    for shard_index in range(args.num_shards):
        shard = torch.load(
            get_embedding_file_name(args.embeddings_dir, args.encoder_ckpt, corpus_file, shard_index=shard_index)
        )
        psg_embs.append(shard)
    torch.save(
        torch.concatenate(psg_embs, dim=0), get_embedding_file_name(args.output_dir, args.encoder_ckpt, corpus_file)
    )

    logger.info("Collected %d shards into %d samples.", args.num_shards, psg_embs.shape[0])


if __name__ == "__main__":
    main()
