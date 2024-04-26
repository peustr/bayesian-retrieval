import json
import logging
import time

import numpy as np
from pytrec_eval import RelevanceEvaluator

from bret.models import BayesianBERTRetriever

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, model, index, device, metrics={"ndcg", "map", "recip_rank"}):
        self.model = model
        self.index = index
        self.device = device
        self.metrics = metrics

    def evaluate_retriever(self, qry_data_loader, qrels, k=10, num_samples=None, run_file=None):
        logger.info("Generating run...")
        t_start = time.time()
        run = self._generate_run(qry_data_loader, k=k, num_samples=num_samples)
        t_end = time.time()
        logger.info("Run generated in %.2f minutes.", (t_end - t_start) / 60)
        if run_file is not None:
            with open(run_file, "w", encoding="utf-8") as f:
                json.dump(run, f)
            logger.info("Run saved in: %s", run_file)
        logger.info("Calculating metrics...")
        results = self._calculate_metrics(run, qrels, k=k)
        return results

    def _generate_run(self, qry_data_loader, k=10, num_samples=None):
        run = {}
        for qry_id, qry in qry_data_loader:
            qry = qry.to(self.device)
            if isinstance(self.model, BayesianBERTRetriever):
                qry_reps, _ = self.model(qry, None, num_samples)
                qry_reps = qry_reps.mean(1)
            else:
                qry_reps, _ = self.model(qry, None)
            scores, indices = self.index.search(qry_reps, k)
            qid = str(qry_id[0])
            run[qid] = {}
            for score, psg_id in zip(scores[0], indices[0]):
                run[qid][str(psg_id)] = float(score)
        return run

    def _calculate_metrics(self, run, qrels, k=10):
        evaluator = RelevanceEvaluator(qrels, self.metrics)
        results = evaluator.evaluate(run)
        map_at_k = []
        mrr_at_k = []
        ndcg_at_k = []
        for _, metrics in results.items():
            map_at_k.append(metrics["map"])
            mrr_at_k.append(metrics["recip_rank"])
            ndcg_at_k.append(metrics["ndcg"])
        results_agg = {
            "MAP@{}".format(k): float(np.mean(map_at_k)),
            "MRR@{}".format(k): float(np.mean(mrr_at_k)),
            "nDCG@{}".format(k): float(np.mean(ndcg_at_k)),
        }
        logger.info(results_agg)
        return results_agg
