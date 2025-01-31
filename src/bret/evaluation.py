import json
import logging
import os
import time

import numpy as np
import torch
from pytrec_eval import RelevanceEvaluator

from bret.encoding import encode_query_mean

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, tokenizer, model, method, device, index=None, metrics={"ndcg", "recip_rank"}):
        self.tokenizer = tokenizer
        self.model = model
        self.method = method
        self.device = device
        self.index = index
        self.metrics = metrics

    def evaluate_retriever(self, qry_data_loader, qrels, k=20, num_samples=None, max_qry_len=32, run_file=None):
        if run_file is not None and os.path.exists(run_file) and os.path.isfile(run_file):
            logger.info("Loading run from: %s", run_file)
            with open(run_file, "r", encoding="utf-8") as f:
                run = json.loads(f.read())
        else:
            logger.info("Generating run...")
            t_start = time.time()
            with torch.no_grad():
                run = self._generate_run(qry_data_loader, k=k, num_samples=num_samples, max_qry_len=max_qry_len)
            t_end = time.time()
            logger.info("Run generated in %.2f minutes.", (t_end - t_start) / 60)
            if run_file is not None:
                with open(run_file, "w", encoding="utf-8") as f:
                    json.dump(run, f)
                logger.info("Run saved in: %s", run_file)
        logger.info("Calculating metrics...")
        results = self._calculate_metrics(run, qrels, k=k)
        return results

    def _generate_run(self, qry_data_loader, k=20, num_samples=None, max_qry_len=32):
        run = {}
        for qry_id, qry in qry_data_loader:
            qry = self.tokenizer(
                qry, padding="max_length", truncation=True, max_length=max_qry_len, return_tensors="pt"
            )
            qry = qry.to(self.device)
            if self.method == "bret":
                qry_reps = self.model(qry, num_samples=num_samples)
                qry_reps = encode_query_mean(qry_reps)
            else:
                qry_reps = self.model(qry)
            scores, indices = self.index.search(qry_reps, k)
            qid = str(qry_id[0])
            run[qid] = {}
            for score, psg_id in zip(scores[0], indices[0]):
                run[qid][str(psg_id)] = float(score)
        return run

    def _calculate_metrics(self, run, qrels, k=20):
        evaluator = RelevanceEvaluator(qrels, self.metrics)
        results = evaluator.evaluate(run)
        ndcg_at_k = []
        mrr_at_k = []
        for _, metrics in results.items():
            ndcg_at_k.append(metrics["ndcg"])
            mrr_at_k.append(metrics["recip_rank"])
        results_agg = {
            "nDCG@{}".format(k): float(np.mean(ndcg_at_k)),
            "MRR@{}".format(k): float(np.mean(mrr_at_k)),
        }
        logger.info(results_agg)
        return results_agg
