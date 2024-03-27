import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from bret.layers.linear import BayesianLinear


class BERTRetriever(nn.Module):
    def __init__(self, backbone, device="cpu"):
        super().__init__()
        self.backbone = backbone
        self.to(device)

    def forward(self, query=None, passage=None):
        qry_reps = self._encode_query(query)
        psg_reps = self._encode_passage(passage)
        return (qry_reps, psg_reps)

    def _encode_query(self, qry):
        if qry is None:
            return None
        qry_out = self.backbone(**qry, return_dict=True)
        qry_reps = qry_out.last_hidden_state[:, 0]
        return qry_reps

    def _encode_passage(self, psg):
        if psg is None:
            return None
        psg_out = self.backbone(**psg, return_dict=True)
        psg_reps = psg_out.last_hidden_state[:, 0]
        return psg_reps

    @classmethod
    def build(cls, model_name, device="cpu", **hf_kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        backbone = AutoModel.from_pretrained(model_name, **hf_kwargs)
        return tokenizer, cls(backbone, device=device)


class BayesianBERTRetriever(BERTRetriever):
    def __init__(self, backbone, device="cpu"):
        super().__init__(backbone, device)
        for i in range(len(backbone.encoder.layer)):
            self.backbone.encoder.layer[i].attention.self.query = BayesianLinear(
                self.backbone.encoder.layer[i].attention.self.query
            )
            self.backbone.encoder.layer[i].attention.self.key = BayesianLinear(
                self.backbone.encoder.layer[i].attention.self.key
            )
            self.backbone.encoder.layer[i].attention.self.value = BayesianLinear(
                self.backbone.encoder.layer[i].attention.self.value
            )
            self.backbone.encoder.layer[i].attention.output.dense = BayesianLinear(
                self.backbone.encoder.layer[i].attention.output.dense
            )
            self.backbone.encoder.layer[i].intermediate.dense = BayesianLinear(
                self.backbone.encoder.layer[i].intermediate.dense
            )
            self.backbone.encoder.layer[i].output.dense = BayesianLinear(self.backbone.encoder.layer[i].output.dense)
        self.backbone.pooler.dense = BayesianLinear(self.backbone.pooler.dense)

    def kl(self):
        sum_kld = None
        for n, m in self.backbone.named_modules():
            if type(m) == BayesianLinear:
                if sum_kld is None:
                    sum_kld = m.kl()
                else:
                    sum_kld += m.kl()
        return sum_kld
