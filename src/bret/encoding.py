import torch

from bret.models.bayesian import BayesianRetriever


def encode_query_mean(qry_reps):
    qry_mean = qry_reps.mean(dim=1)
    return qry_mean


def encode_passage_mean(psg_reps):
    psg_mean = psg_reps.mean(dim=1)
    return psg_mean


def encode_query_multivariate(qry_reps):
    qry_mean = qry_reps.mean(dim=1)
    qry_var = qry_reps.var(dim=1)
    batch_size = qry_mean.size(0)
    k = qry_mean.size(1)
    rep = torch.zeros(batch_size, 3 * k + 3, device=qry_mean.device)
    rep[:, 0] = 1
    rep[:, 1] = torch.log(qry_var).sum(1)
    rep[:, 2 : k + 2] = qry_var
    rep[:, k + 2 : 2 * k + 2] = qry_mean**2
    rep[:, 2 * k + 2 : -1] = qry_mean
    rep[:, -1] = k
    return 0.5 * rep


def encode_passage_multivariate(psg_reps):
    psg_mean = psg_reps.mean(dim=1)
    psg_var = psg_reps.var(dim=1)
    batch_size = psg_mean.size(0)
    k = psg_mean.size(1)
    rep = torch.zeros(batch_size, 3 * k + 3, device=psg_mean.device)
    rep[:, 0] = -(torch.log(psg_var) + psg_mean**2 / psg_var).sum(1)
    rep[:, 1] = 1
    rep[:, 2 : k + 2] = -1 / psg_var
    rep[:, k + 2 : 2 * k + 2] = -1 / psg_var
    rep[:, 2 * k + 2 : -1] = (2 * psg_mean) / psg_var
    rep[:, -1] = 1
    return rep


def encode_queries(queries, encoder, device, num_samples=None):
    qry_embs = []
    with torch.no_grad():
        for _, qry in queries:
            qry = qry.to(device)
            if isinstance(encoder, BayesianRetriever):
                qry_reps, _ = encoder(qry, None, num_samples=num_samples)
                qry_reps = encode_query_mean(qry_reps)
            else:
                qry_reps, _ = encoder(qry, None)
            qry_embs.append(qry_reps.detach().cpu())
    qry_embs = torch.cat(qry_embs, dim=0)
    return qry_embs


def encode_corpus(corpus, encoder, device, num_samples=None):
    psg_embs = []
    with torch.no_grad():
        for _, psg in corpus:
            psg = psg.to(device)
            if isinstance(encoder, BayesianRetriever):
                _, psg_reps = encoder(None, psg, num_samples=num_samples)
                psg_reps = encode_passage_mean(psg_reps)
            else:
                _, psg_reps = encoder(None, psg)
            psg_embs.append(psg_reps.detach().cpu())
    psg_embs = torch.cat(psg_embs, dim=0)
    return psg_embs
