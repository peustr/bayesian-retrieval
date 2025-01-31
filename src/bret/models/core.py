import logging

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from bret.utils import disable_grad

logger = logging.getLogger(__name__)


class Retriever(nn.Module):
    def __init__(self, backbone, device="cpu"):
        super().__init__()
        self.backbone = backbone
        self.device = device
        self.to(device)

    def forward(self, qry_or_psg):
        return self._encode(qry_or_psg)

    def _encode(self, qry_or_psg):
        model_output = self.backbone(**qry_or_psg, return_dict=True)
        embeddings = self.cls_pooling(model_output, qry_or_psg["attention_mask"])
        return embeddings

    @classmethod
    def build(cls, model_name, device="cpu", **hf_kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        backbone = AutoModel.from_pretrained(model_name, **hf_kwargs)
        return tokenizer, cls(backbone, device=device)


class BERTRetriever(Retriever):
    def __init__(self, backbone, device="cpu"):
        super().__init__(backbone, device)
        disable_grad(self.backbone.embeddings)

    def cls_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class DistilBERTRetriever(Retriever):
    def __init__(self, backbone, device="cpu"):
        super().__init__(backbone, device)
        disable_grad(self.backbone.embeddings)

    def cls_pooling(self, model_output, *args):
        return model_output.last_hidden_state[:, 0]
