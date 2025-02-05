import torch

from bret.layers.linear import BayesianLinear
from bret.models.core import Retriever
from bret.utils import disable_grad


class BayesianRetriever(Retriever):
    def __init__(self, backbone, device="cpu"):
        super().__init__(backbone, device)

    def kl(self):
        sum_kld = torch.tensor(0.0, device=self.device)
        for _, m in self.backbone.named_modules():
            if isinstance(m, BayesianLinear):
                sum_kld += m.kl()
        return sum_kld

    def forward(self, qry_or_psg, num_samples=None):
        if num_samples is None or num_samples == 1:
            return self._encode(qry_or_psg)
        # Note: we sample multiple times during inference, so we do not care that the posterior is overwritten.
        embs = []
        for _ in range(num_samples):
            embs.append(self._encode(qry_or_psg))
        return torch.stack(embs)


class BayesianBERTRetriever(BayesianRetriever):
    def __init__(self, backbone, device="cpu"):
        super().__init__(backbone, device)
        disable_grad(self.backbone.embeddings)
        for i in range(len(backbone.encoder.layer)):
            self.backbone.encoder.layer[i].intermediate.dense = BayesianLinear(
                self.backbone.encoder.layer[i].intermediate.dense,
            )
            self.backbone.encoder.layer[i].output.dense = BayesianLinear(
                self.backbone.encoder.layer[i].output.dense,
            )

    def cls_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class BayesianDistilBERTRetriever(BayesianRetriever):
    def __init__(self, backbone, device="cpu"):
        super().__init__(backbone, device)
        disable_grad(self.backbone.embeddings)
        for i in range(len(backbone.transformer.layer)):
            self.backbone.transformer.layer[i].ffn.lin1 = BayesianLinear(
                self.backbone.transformer.layer[i].ffn.lin1,
            )
            self.backbone.transformer.layer[i].ffn.lin2 = BayesianLinear(
                self.backbone.transformer.layer[i].ffn.lin2,
            )

    def cls_pooling(self, model_output, *args):
        return model_output.last_hidden_state[:, 0]
