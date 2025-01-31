import torch

from bret.models.bayesian import BayesianRetriever


def encode_query_mean(qry_reps):
    return qry_reps.mean(dim=0)


def encode_passage_mean(psg_reps):
    return psg_reps.mean(dim=0)


def encode_queries(queries, tokenizer, encoder, device, method, num_samples=None, max_qry_len=32):
    qry_embs = []
    with torch.no_grad():
        for _, qry in queries:
            qry = tokenizer(qry, padding="max_length", truncation=True, max_length=max_qry_len, return_tensors="pt")
            qry = qry.to(device)
            if method == "bret":
                qry_reps = encoder(qry, num_samples=num_samples)
                qry_reps = encode_query_mean(qry_reps)
            else:
                qry_reps = encoder(qry)
            qry_embs.append(qry_reps.detach().cpu())
        qry_embs = torch.cat(qry_embs, dim=0)
    return qry_embs


def encode_corpus(corpus, tokenizer, encoder, device, method, num_samples=None, max_psg_len=256):
    psg_embs = []
    with torch.no_grad():
        for _, psg in corpus:
            psg = tokenizer(psg, padding="max_length", truncation=True, max_length=max_psg_len, return_tensors="pt")
            psg = psg.to(device)
            if method == "bret":
                psg_reps = encoder(psg, num_samples=num_samples)
                psg_reps = encode_passage_mean(psg_reps)
            else:
                psg_reps = encoder(psg)
            psg_embs.append(psg_reps.detach().cpu())
        psg_embs = torch.cat(psg_embs, dim=0)
    return psg_embs
