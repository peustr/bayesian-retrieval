import logging

import numpy as np
from pytrec_eval import RelevanceEvaluator

logger = logging.getLogger(__name__)


def generate_run(data_loader, model, index, device, k=20, method=None, num_samples=None):
    run = {}
    for qry_id, qry in data_loader:
        qry = qry.to(device)
        if method == "vi":
            qry_reps, _ = model(qry, None, num_samples)
            qry_reps = qry_reps.mean(1)
        else:
            qry_reps, _ = model(qry, None)
        scores, indices = index.search(qry_reps, k)
        qid = str(qry_id[0])
        run[qid] = {}
        for score, psg_id in zip(scores[0], indices[0]):
            run[qid][str(psg_id)] = float(score)
        return run


def calculate_metrics(run, qrels, metrics={"ndcg", "map", "recip_rank"}, k=20):
    evaluator = RelevanceEvaluator(qrels, metrics)
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
