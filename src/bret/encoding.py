import torch

from bret.models.bayesian import BayesianRetriever


def encode_query_mean(qry_reps):
    return qry_reps.mean(dim=0)


def encode_passage_mean(psg_reps):
    return psg_reps.mean(dim=0)


def encode_queries(queries, encoder, device, method, num_samples=None):
    qry_embs = []
    with torch.no_grad():
        for _, qry in queries:
            qry = qry.to(device)
            if method == "bret":
                qry_reps, _ = encoder(qry, None, num_samples=num_samples)
                qry_reps = encode_query_mean(qry_reps)
            else:
                qry_reps, _ = encoder(qry, None)
            qry_embs.append(qry_reps.detach().cpu())
        qry_embs = torch.cat(qry_embs, dim=0)
    return qry_embs


def encode_corpus(corpus, encoder, device, method, num_samples=None):
    psg_embs = []
    with torch.no_grad():
        for _, psg in corpus:
            psg = psg.to(device)
            if method == "bret":
                _, psg_reps = encoder(None, psg, num_samples=num_samples)
                psg_reps = encode_passage_mean(psg_reps)
            else:
                _, psg_reps = encoder(None, psg)
            psg_embs.append(psg_reps.detach().cpu())
        psg_embs = torch.cat(psg_embs, dim=0)
    return psg_embs
