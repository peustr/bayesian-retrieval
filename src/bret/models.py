import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from bret.layers.linear import BayesianLinear


class BERTRetriever(nn.Module):
    def __init__(self, backbone, device="cpu"):
        super().__init__()
        self.backbone = backbone
        self.to(device)

    def forward(self, query=None, passage=None):
        qry_reps = self._encode(query)
        psg_reps = self._encode(passage)
        return (qry_reps, psg_reps)

    def _encode(self, qry_or_psg):
        if qry_or_psg is None:
            return None
        out = self.backbone(**qry_or_psg, return_dict=True)
        return out.last_hidden_state[:, 0]

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
        for _, m in self.backbone.named_modules():
            if type(m) == BayesianLinear:
                if sum_kld is None:
                    sum_kld = m.kl()
                else:
                    sum_kld += m.kl()
        return sum_kld

    def forward(self, query=None, passage=None, num_samples=None):
        if num_samples is None:
            qry_reps = self._encode(query)
            if qry_reps is not None:
                qry_reps = qry_reps.unsqueeze(1)
            psg_reps = self._encode(passage)
            if psg_reps is not None:
                psg_reps = psg_reps.unsqueeze(1)
            return (qry_reps, psg_reps)

        qry_reps = self._encode_vi(query, num_samples)
        psg_reps = self._encode_vi(passage, num_samples)

        return (qry_reps, psg_reps)

    def _encode_vi(self, qry_or_psg, num_samples):
        reps = None
        if qry_or_psg is not None:
            batch_size = qry_or_psg["input_ids"].size(0)
            qry_or_psg["input_ids"] = qry_or_psg["input_ids"].repeat_interleave(num_samples, dim=0)
            qry_or_psg["attention_mask"] = qry_or_psg["attention_mask"].repeat_interleave(num_samples, dim=0)
            reps = self._encode(qry_or_psg).reshape(batch_size, num_samples, -1)
        return reps
