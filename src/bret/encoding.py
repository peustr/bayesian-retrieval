import torch

from bret.models import BayesianBERTRetriever


def encode_corpus(corpus_data, encoder, device, num_samples=None):
    psg_embs = []
    for _, psg in corpus_data:
        psg = psg.to(device)
        if isinstance(encoder, BayesianBERTRetriever):
            _, psg_reps = encoder(None, psg, num_samples)
            psg_reps = psg_reps.mean(1)
        else:
            _, psg_reps = encoder(None, psg)
        psg_embs.append(psg_reps.detach().cpu())
    psg_embs = torch.cat(psg_embs, dim=0)
    return psg_embs
