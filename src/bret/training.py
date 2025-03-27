import logging
import time
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from bret.encoding import encode_corpus
from bret.evaluation import Evaluator
from bret.indexing import FaissIndex
from bret.losses import BinaryPassageRetrievalLoss
from bret.models.mc_dropout import MCDropoutRetriever

logger = logging.getLogger(__name__)


def make_lr_scheduler_with_warmup(model, training_data, lr, min_lr, num_epochs, warmup_rate):
    optimizer = Adam(model.parameters(), lr=lr)
    num_training_steps = len(training_data) * num_epochs
    warmup_iters = int(warmup_rate * num_training_steps)
    decay_iters = int((1 - warmup_rate) * num_training_steps)
    decay_factor = min_lr / lr
    warmup = LinearLR(optimizer, start_factor=decay_factor, end_factor=1.0, total_iters=warmup_iters)
    decay = LinearLR(optimizer, start_factor=1.0, end_factor=decay_factor, total_iters=decay_iters)
    scheduler = SequentialLR(optimizer, [warmup, decay], [warmup_iters])
    logger.info("Using linear learning rate scheduling with linear warm-up.")
    logger.info(
        "Total training steps: %d | LR warm-up for %d steps. | LR decay for %d steps.",
        num_training_steps,
        warmup_iters,
        decay_iters,
    )
    return optimizer, scheduler


class DPRTrainer:
    def __init__(self, tokenizer, model, training_data, validation_queries, validation_corpus, qrels, device):
        self.tokenizer = tokenizer
        self.model = model
        self.training_data = training_data
        self.validation_queries = validation_queries
        self.validation_corpus = validation_corpus
        self.qrels = qrels
        self.device = device
        self.loss_func = BinaryPassageRetrievalLoss()

    def prepare_inputs(self, qry, pos_psg, neg_psg, max_qry_len, max_psg_len):
        qry_enc = self.tokenizer(
            qry, padding="max_length", truncation=True, max_length=max_qry_len, return_tensors="pt"
        ).to(self.device)
        pos_enc = self.tokenizer(
            pos_psg, padding="max_length", truncation=True, max_length=max_psg_len, return_tensors="pt"
        ).to(self.device)
        neg_enc = self.tokenizer(
            neg_psg, padding="max_length", truncation=True, max_length=max_psg_len, return_tensors="pt"
        ).to(self.device)
        return qry_enc, pos_enc, neg_enc

    def compute_loss(self, optimizer, qry_enc, pos_enc, neg_enc, **kwargs) -> Dict[str, torch.Tensor]:
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=True):
            optimizer.zero_grad()
            qry_emb = self.model(qry_enc)
            pos_emb = self.model(pos_enc)
            neg_emb = self.model(neg_enc)
            return {"loss": self.loss_func(qry_emb, pos_emb, neg_emb)}

    def compute_validation_metrics(self, k, **kwargs):
        return self._compute_validation_metrics("dpr", k=k)

    def train(
            self,
            num_epochs=4,
            lr=5e-6,
            min_lr=5e-8,
            warmup_rate=0.1,
            ckpt_file_name=None,
            k=20,
            max_qry_len=32,
            max_psg_len=256,
            log_frequency=1,
            loss_kwargs=None,
            **kwargs,
    ):
        optimizer, scheduler = make_lr_scheduler_with_warmup(
            self.model, self.training_data, lr, min_lr, num_epochs, warmup_rate
        )
        if ckpt_file_name is not None:
            max_ndcg_at_k = 0.0
        else:
            max_ndcg_at_k = 1.0

        loss_kwargs = loss_kwargs or {}
        logger.info(f"loss_kwargs: {loss_kwargs}")
        scaler = torch.amp.GradScaler(enabled=True)
        for epoch in range(1, num_epochs + 1):
            t_start = time.time()
            self.model.train()
            losses = defaultdict(list)
            for batch_num, (qry, pos_psg, neg_psg) in enumerate(self.training_data):
                qry_enc, pos_enc, neg_enc = self.prepare_inputs(qry=qry,
                                                                pos_psg=pos_psg,
                                                                neg_psg=neg_psg,
                                                                max_qry_len=max_qry_len,
                                                                max_psg_len=max_psg_len)
                loss_terms = self.compute_loss(optimizer,
                                               qry_enc=qry_enc,
                                               pos_enc=pos_enc,
                                               neg_enc=neg_enc,
                                               **loss_kwargs)
                loss = loss_terms["loss"]
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                for loss_name, loss_value in loss_terms.items():
                    losses[loss_name].append(loss_value.item())
                    if batch_num % log_frequency == 0:
                        logger.info(f"epoch={epoch}::Batch:{batch_num}:: {loss_name}={loss_value.item()}")

            metrics = self.compute_validation_metrics(k=k, **kwargs)
            ndcg_at_k = metrics["nDCG@" + str(k)]
            mrr_at_k = metrics["MRR@" + str(k)]
            t_end = time.time()
            logger.info("Epoch %d finished in %.2f minutes.", epoch, (t_end - t_start) / 60)
            for loss_name, loss_value in losses.items():
                logger.info(f"{loss_name}={np.mean(loss_value):.3f}")
            logger.info("Validation metrics: nDCG@%d=%.3f | MRR@%d=%.3f", k, ndcg_at_k, k, mrr_at_k)
            if ndcg_at_k > max_ndcg_at_k:
                torch.save(self.model.state_dict(), ckpt_file_name)
                max_ndcg_at_k = ndcg_at_k
                logger.info("Model saved in: %s", ckpt_file_name)

    def _compute_validation_metrics(self, method, k=20, **kwargs):
        self.model.eval()
        psg_embs = encode_corpus(self.validation_corpus, self.tokenizer, self.model, self.device, method, **kwargs)
        index = FaissIndex.build(psg_embs)
        evaluator = Evaluator(
            self.tokenizer,
            self.model,
            method,
            self.device,
            index=index,
            metrics={"ndcg", "recip_rank"},
        )
        return evaluator.evaluate_retriever(self.validation_queries, self.qrels, k=k, **kwargs)


class BayesianDPRTrainer(DPRTrainer):
    def __init__(self, tokenizer, model, training_data, validation_queries, validation_corpus, qrels, device):
        super().__init__(tokenizer, model, training_data, validation_queries, validation_corpus, qrels, device)

    def compute_loss(self, optimizer, qry_enc, pos_enc, neg_enc, **kwargs) -> Dict[str, torch.Tensor]:
        assert "kld_weight" in kwargs, "provide kld_weight in loss_kwargs!"
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=True):
            optimizer.zero_grad()
            qry_emb = self.model(qry_enc)
            pos_emb = self.model(pos_enc, use_cached_posterior=True)
            neg_emb = self.model(neg_enc, use_cached_posterior=True)
            loss_ce = self.loss_func(qry_emb, pos_emb, neg_emb)
            loss_kld = self.model.kl() / len(self.training_data.dataset)
            loss = loss_ce + kwargs["kld_weight"] * loss_kld
            return {"loss": loss, "loss_ce": loss_ce, "loss_kld": loss_kld}

    def compute_validation_metrics(self, k, **kwargs):
        return self._compute_validation_metrics("bret", k=k, num_samples=kwargs["num_samples"])


class MCDropoutDPRTrainer(DPRTrainer):
    def compute_loss(self, optimizer, qry_enc, pos_enc, neg_enc, **kwargs) -> Dict[str, torch.Tensor]:
        self.model: MCDropoutRetriever
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=True):
            optimizer.zero_grad()
            qry_emb = self.model(qry_enc)
            reg_term = self.model.regularization
            pos_emb = self.model(pos_enc)
            reg_term += self.model.regularization
            neg_emb = self.model(neg_enc)
            reg_term += self.model.regularization

            loss_ce = self.loss_func(qry_emb, pos_emb, neg_emb)
            loss_reg = reg_term.squeeze()
            loss = loss_ce + loss_reg
            return {"loss": loss, "loss_ce": loss_ce, "loss_reg": loss_reg}
