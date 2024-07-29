import logging

import faiss
import numpy as np
import torch

logger = logging.getLogger(__name__)


def _ensure_numpy(rep):
    if isinstance(rep, torch.Tensor):
        return rep.detach().cpu().numpy()
    return rep


class FaissIndex:
    def __init__(self, rep_dim):
        index = faiss.IndexFlatIP(rep_dim)
        self.index = index

    def add(self, psg_reps):
        psg_reps = _ensure_numpy(psg_reps)
        self.index.add(psg_reps)

    def search(self, qry_rep, k):
        qry_rep = _ensure_numpy(qry_rep)
        return self.index.search(qry_rep, k)

    def batch_search(self, qry_reps, k, batch_size):
        qry_reps = _ensure_numpy(qry_reps)
        num_queries = qry_reps.shape[0]
        scores = []
        indices = []
        for i in range(0, num_queries, batch_size):
            batch_scores, batch_indices = self.search(qry_reps[i : i + batch_size], k)
            scores.append(batch_scores)
            indices.append(batch_indices)
        scores = np.concatenate(scores, axis=0)
        indices = np.concatenate(indices, axis=0)
        return scores, indices

    @classmethod
    def build(cls, embeddings):
        index = cls(embeddings.size(1))
        index.add(embeddings)
        return index
