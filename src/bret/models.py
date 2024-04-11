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
        for _, m in self.backbone.named_modules():
            if type(m) == BayesianLinear:
                if sum_kld is None:
                    sum_kld = m.kl()
                else:
                    sum_kld += m.kl()
        return sum_kld

    def forward(self, query=None, passage=None, num_samples=None):
        if num_samples is None:
            qry_reps = self._encode_query(query)
            if qry_reps is not None:
                qry_reps = qry_reps.unsqueeze(1)
            psg_reps = self._encode_passage(passage)
            if psg_reps is not None:
                psg_reps = psg_reps.unsqueeze(1)
            return (qry_reps, psg_reps)

        assert query["input_ids"].size(1) == passage["input_ids"].size(1)
        feature_dim = query["input_ids"].size(1)

        num_queries = query["input_ids"].size(0)
        query["input_ids"] = query["input_ids"].repeat_interleave(num_samples, dim=0)
        query["attention_mask"] = query["attention_mask"].repeat_interleave(num_samples, dim=0)
        qry_reps = self._encode_query(query)
        qry_reps = qry_reps.reshape(num_queries, num_samples, feature_dim)

        num_passages = passage["input_ids"].size(0)
        passage["input_ids"] = passage["input_ids"].repeat_interleave(num_samples, dim=0)
        passage["attention_mask"] = passage["attention_mask"].repeat_interleave(num_samples, dim=0)
        psg_reps = self._encode_passage(passage)
        psg_reps = psg_reps.reshape(num_passages, num_samples, feature_dim)

        return (qry_reps, psg_reps)
