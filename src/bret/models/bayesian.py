import torch

from bret.layers.linear import BayesianLinear
from bret.models.core import Retriever


class BayesianRetriever(Retriever):
    def __init__(self, backbone, device="cpu"):
        super().__init__(backbone, device)

    def kl(self):
        sum_kld = None
        for _, m in self.backbone.named_modules():
            if isinstance(m, BayesianLinear):
                if sum_kld is None:
                    sum_kld = m.kl()
                else:
                    sum_kld += m.kl()
        return sum_kld

    def forward(self, query=None, passage=None, num_samples=None):
        num_samples = num_samples or 1
        if num_samples == 1:
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


class BayesianBERTRetriever(BayesianRetriever):
    def __init__(self, backbone, device="cpu"):
        super().__init__(backbone, device)
        for i in range(len(backbone.encoder.layer)):
            self.backbone.encoder.layer[i].intermediate.dense = BayesianLinear(
                self.backbone.encoder.layer[i].intermediate.dense
            )
            self.backbone.encoder.layer[i].output.dense = BayesianLinear(self.backbone.encoder.layer[i].output.dense)

    def cls_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class BayesianDistilBERTRetriever(BayesianRetriever):
    def __init__(self, backbone, device="cpu"):
        super().__init__(backbone, device)
        for i in range(len(backbone.transformer.layer)):
            self.backbone.transformer.layer[i].ffn.lin1 = BayesianLinear(self.backbone.transformer.layer[i].ffn.lin1)
            self.backbone.transformer.layer[i].ffn.lin2 = BayesianLinear(self.backbone.transformer.layer[i].ffn.lin2)

    def cls_pooling(self, model_output, *args):
        return model_output.last_hidden_state[:, 0]
