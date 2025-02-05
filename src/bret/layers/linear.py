import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal


class BayesianLinear(nn.Module):
    def __init__(self, prior, prior_scale=0.01):
        super().__init__()
        self.prior = Normal(prior.weight.data.detach().clone(), prior_scale)
        self.weight_mean = prior.weight
        self.weight_logvar = nn.Parameter(-np.log(2**16) + 0.5 * torch.randn_like(self.weight_mean))
        self.bias = prior.bias
        if self.bias is not None:
            # We don't have to sample/train the bias, as it's just an offset to the mean.
            self.bias.requires_grad = False

    @property
    def weight_var(self):
        return self.weight_logvar.exp()

    def kl(self):
        if hasattr(self, "posterior"):
            return kl_divergence(self.posterior, self.prior).sum()
        raise AttributeError("Posterior not set. kl() needs to be called after a forward pass.")

    def forward(self, x, use_cached_sample=False):
        if use_cached_sample and hasattr(self, "_cached_W"):
            W = self._cached_W
        self.posterior = Normal(self.weight_mean, self.weight_var.sqrt())
        W = self.posterior.rsample()
        self._cached_W = W
        return F.linear(x, W, self.bias)
