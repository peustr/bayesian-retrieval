import logging

import torch
from torch import nn

import numpy as np

from bret.models.core import Retriever
from bret.utils.model_utils import get_transformer_hidden_dim, disable_grad


class ConcreteDropout(nn.Module):
    """
    Source: https://github.com/dscohen/LastLayersBayesianIR/blob/main/models/layers/concete_dropout.py#L8
    Concrete Dropout.
    Implementation of the Concrete Dropout module as described in the
    'Concrete Dropout' paper: https://arxiv.org/pdf/1705.07832
    """

    def __init__(self,
                 weight_regulariser: float,
                 dropout_regulariser: float,
                 init_min: float = 0.1,
                 init_max: float = 0.1) -> None:
        """Concrete Dropout.
        Parameters
        ----------
        weight_regulariser : float
            Weight regulariser term.
        dropout_regulariser : float
            Dropout regulariser term.
        init_min : float
            Initial min value.
        init_max : float
            Initial max value.
        """

        super().__init__()

        self.weight_regulariser = weight_regulariser
        self.dropout_regulariser = dropout_regulariser

        init_min = np.log(init_min) - np.log(1.0 - init_min)
        init_max = np.log(init_max) - np.log(1.0 - init_max)

        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))
        self.p = torch.sigmoid(self.p_logit)

        self.regularisation = 0.0

    def forward(self, x: torch.Tensor, layer: nn.Module) -> torch.Tensor:
        """Calculates the forward pass.
        The regularisation term for the layer is calculated and assigned to a
        class attribute - this can later be accessed to evaluate the loss.
        Parameters
        ----------
        x : Tensor
            Input to the Concrete Dropout.
        layer : nn.Module
            Layer for which to calculate the Concrete Dropout.
        Returns
        -------
        Tensor
            Output from the dropout layer.
        """

        output = layer(self._concrete_dropout(x))

        sum_of_squares = 0
        for param in layer.parameters():
            sum_of_squares += torch.sum(torch.pow(param, 2))

        weights_reg = self.weight_regulariser * sum_of_squares / (1.0 - self.p)

        dropout_reg = self.p * torch.log(self.p)
        dropout_reg += (1.0 - self.p) * torch.log(1.0 - self.p)
        dropout_reg *= self.dropout_regulariser * x[0].numel()

        self.regularisation = weights_reg + dropout_reg

        return output

    def _concrete_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the Concrete Dropout.
        Parameters
        ----------
        x : Tensor
            Input tensor to the Concrete Dropout layer.
        Returns
        -------
        Tensor
            Outputs from Concrete Dropout.
        """

        eps = 1e-7
        tmp = 0.1

        self.p = torch.sigmoid(self.p_logit)
        u_noise = torch.rand_like(x)

        drop_prob = (torch.log(self.p + eps) -
                     torch.log(1 - self.p + eps) +
                     torch.log(u_noise + eps) -
                     torch.log(1 - u_noise + eps))

        drop_prob = torch.sigmoid(drop_prob / tmp)

        random_tensor = 1 - drop_prob
        retain_prob = 1 - self.p

        x = torch.mul(x, random_tensor) / retain_prob

        return x


class MCDropoutRetriever(Retriever):
    def __init__(self, backbone, device="cpu"):
        super().__init__(backbone, device)
        self.hidden_dim = get_transformer_hidden_dim(backbone)
        self.stoch_projection_1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.stoch_projection_2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        w, d = 1e-6, 1e-3
        self.cd1 = ConcreteDropout(weight_regulariser=w, dropout_regulariser=d)
        self.cd2 = ConcreteDropout(weight_regulariser=w, dropout_regulariser=d)
        self.regularization = None

    def _mc_sample(self, embeds, n_iters):
        out_reps = []
        for _ in range(n_iters):
            out_features = self.cd1(embeds, torch.nn.Sequential(self.stoch_projection_1, nn.ReLU()))
            out_features = self.cd2(out_features, torch.nn.Sequential(self.stoch_projection_2))
            out_reps.append(out_features)

        # size = n_iters * batch_size * 2, not what we want!
        out_reps = torch.stack(out_reps)
        # convert to batch_size * n_iters * 2
        out_reps.swapaxes_(0, 1)
        return out_reps

    def forward(self, qry_or_psg, num_samples=None):
        if num_samples is None or num_samples == 1:
            return self._encode(qry_or_psg)

        self.regularization = self.cd1.regularisation + self.cd2.regularisation
        logging.warning("implement regularisation!!!!!!!!!!")
        reps = self._mc_sample(self._encode(qry_or_psg),
                               n_iters=num_samples)

        return torch.stack(reps)

    def compute_uncertainty(self, qry_or_psg, num_samples):
        if num_samples is None or num_samples == 1:
            raise ValueError("Need multiple samples to compute uncertainty.")
        return self.compute_representation_uncertainty(self.forward(qry_or_psg, num_samples))

    def compute_representation_uncertainty(self, embeddings):
        # embeddings should be of shape (num_samples x batch_size x embedding_dim)
        return embeddings.var(dim=0).sum(dim=1)


class MCDropoutDistilBERTRetriever(MCDropoutRetriever):
    def __init__(self, backbone, device="cpu"):
        super().__init__(backbone, device)
        disable_grad(self.backbone.embeddings)


class MCDropoutBERTRetriever(MCDropoutRetriever):
    def __init__(self, backbone, device="cpu"):
        super().__init__(backbone, device)
        disable_grad(self.backbone.embeddings)
