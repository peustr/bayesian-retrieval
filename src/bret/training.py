import logging
import time

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from bret.encoding import encode_corpus, encode_passage_mean, encode_query_mean
from bret.evaluation import Evaluator
from bret.indexing import FaissIndex
from bret.relevance import dot_product_similarity

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
    def __init__(self, model, training_data, validation_queries, validation_corpus, qrels, device):
        self.model = model
        self.training_data = training_data
        self.validation_queries = validation_queries
        self.validation_corpus = validation_corpus
        self.qrels = qrels
        self.device = device

    def train(self, num_epochs=4, lr=5e-6, min_lr=5e-8, warmup_rate=0.1, ckpt_file_name=None, k=20, **kwargs):
        optimizer, scheduler = make_lr_scheduler_with_warmup(
            self.model, self.training_data, lr, min_lr, num_epochs, warmup_rate
        )
        if ckpt_file_name is not None:
            max_ndcg_at_k = 0.0
        else:
            max_ndcg_at_k = 1.0
        scaler = torch.amp.GradScaler(self.device.type, enabled=True)
        for epoch in range(1, num_epochs + 1):
            t_start = time.time()
            self.model.train()
            for qry, psg in self.training_data:
                qry = qry.to(self.device)
                psg = psg.to(self.device)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=True):
                    qry_reps, psg_reps = self.model(qry, psg)
                    scores = dot_product_similarity(qry_reps, psg_reps)
                    targets = torch.arange(scores.size(0), device=self.device, dtype=torch.long) * (
                        psg_reps.size(0) // qry_reps.size(0)
                    )
                    loss = F.cross_entropy(scores, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            metrics = self._compute_validation_metrics("dpr", k=k)
            ndcg_at_k = metrics["nDCG@" + str(k)]
            mrr_at_k = metrics["MRR@" + str(k)]
            t_end = time.time()
            logger.info("Epoch %d finished in %.2f minutes.", epoch, (t_end - t_start) / 60)
            logger.info("Validation metrics: nDCG@%d=%.3f | MRR@%d=%.3f", k, ndcg_at_k, k, mrr_at_k)
            if ndcg_at_k > max_ndcg_at_k:
                torch.save(self.model.state_dict(), ckpt_file_name)
                max_ndcg_at_k = ndcg_at_k
                logger.info("Model saved in: %s", ckpt_file_name)

    def _compute_validation_metrics(self, method, k=20, **kwargs):
        self.model.eval()
        psg_embs = encode_corpus(self.validation_corpus, self.model, self.device, method, **kwargs)
        index = FaissIndex.build(psg_embs)
        evaluator = Evaluator(
            self.model,
            method,
            self.device,
            index=index,
            metrics={"ndcg", "recip_rank"},
        )
        return evaluator.evaluate_retriever(self.validation_queries, self.qrels, k=k, **kwargs)


class BayesianDPRTrainer(DPRTrainer):
    def __init__(self, model, training_data, validation_queries, validation_corpus, qrels, device):
        super().__init__(model, training_data, validation_queries, validation_corpus, qrels, device)

    def train(self, num_epochs=4, lr=5e-6, min_lr=5e-8, warmup_rate=0.1, ckpt_file_name=None, k=20, num_samples=10):
        optimizer, scheduler = make_lr_scheduler_with_warmup(
            self.model, self.training_data, lr, min_lr, num_epochs, warmup_rate
        )
        if ckpt_file_name is not None:
            max_ndcg_at_k = 0.0
        else:
            max_ndcg_at_k = 1.0
        scaler = torch.amp.GradScaler(self.device.type, enabled=True)
        for epoch in range(1, num_epochs + 1):
            t_start = time.time()
            self.model.train()
            for qry, psg in self.training_data:
                qry = qry.to(self.device)
                psg = psg.to(self.device)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=True):
                    qry_reps, psg_reps = self.model(qry, psg, num_samples=num_samples)
                    qry_reps = encode_query_mean(qry_reps)
                    psg_reps = encode_passage_mean(psg_reps)
                    scores = dot_product_similarity(qry_reps, psg_reps)
                    targets = torch.arange(scores.size(0), device=self.device, dtype=torch.long) * (
                        psg_reps.size(0) // qry_reps.size(0)
                    )
                    loss_ce = F.cross_entropy(scores, targets)
                    loss_kld = self.model.kl() / len(self.training_data.dataset)
                    loss = loss_ce + loss_kld
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            metrics = self._compute_validation_metrics("bret", k=k, num_samples=num_samples)
            ndcg_at_k = metrics["nDCG@" + str(k)]
            mrr_at_k = metrics["MRR@" + str(k)]
            t_end = time.time()
            logger.info("Epoch %d finished in %.2f minutes.", epoch, (t_end - t_start) / 60)
            logger.info("Validation metrics: nDCG@%d=%.3f | MRR@%d=%.3f", k, ndcg_at_k, k, mrr_at_k)
            if ndcg_at_k > max_ndcg_at_k:
                torch.save(self.model.state_dict(), ckpt_file_name)
                max_ndcg_at_k = ndcg_at_k
                logger.info("Model saved in: %s", ckpt_file_name)
