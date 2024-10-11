import numpy as np
import torch
import torch.nn as nn


class BayesianLinear(nn.Module):
    def __init__(self, prior, prior_var=1e-3):
        super().__init__()
        self.weight_prior_mean = prior.weight.data.clone().detach()
        self.weight_prior_var = prior_var
        self.weight_mean = prior.weight
        self._init_logvar()
        self.bias = prior.bias
        if self.bias is not None:
            # We don't have to sample/train the bias, as it's just an offset to the mean.
            self.bias.requires_grad = False

    def _init_logvar(self):
        self.weight_logvar = nn.Parameter(-np.log(2**10) + 0.5 * torch.randn_like(self.weight_mean))

    @property
    def weight_var(self):
        return self.weight_logvar.exp()

    def kl(self):
        return 0.5 * (
            (((self.weight_mean - self.weight_prior_mean).pow(2) + self.weight_var) / self.weight_prior_var).sum()
            - (self.weight_var / self.weight_prior_var).log().sum()
            - self.weight_mean.numel()
        )

    def forward(self, x):
        batch_size = x.size(0)
        z = (
            torch.randn(
                (batch_size, *self.weight_mean.shape), dtype=self.weight_mean.dtype, device=self.weight_mean.device
            )
            * self.weight_var.sqrt()
        )
        if self.bias is not None:
            return torch.bmm(x, (self.weight_mean + z).transpose(-2, -1)) + self.bias
        return torch.bmm(x, (self.weight_mean + z).transpose(-2, -1))
