import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal


class BayesianLinear(nn.Module):
    def __init__(self, prior, weight_prior_scale=0.01, bias_prior_scale=0.01):
        super().__init__()
        #
        self.weight_prior = Normal(prior.weight.data.detach().clone(), weight_prior_scale)
        self.weight_mean = prior.weight
        self.weight_logvar = nn.Parameter(-np.log(2**16) + 0.5 * torch.randn_like(self.weight_mean))
        #
        self.bias_prior = Normal(prior.bias.data.detach().clone(), bias_prior_scale)
        self.bias_mean = prior.bias
        self.bias_logvar = nn.Parameter(-np.log(2**16) + 0.5 * torch.randn_like(self.bias_mean))
        #
        self._use_cached_posterior = False

    @property
    def weight_var(self):
        return self.weight_logvar.exp()

    @property
    def bias_var(self):
        return self.bias_logvar.exp()

    def kl(self, reduction="sum"):
        assert reduction in ("sum", "mean")
        kl_weight = kl_divergence(self.weight_posterior, self.weight_prior)
        kl_bias = kl_divergence(self.bias_posterior, self.bias_prior)
        if hasattr(self, "weight_posterior") and hasattr(self, "bias_posterior"):
            if reduction == "sum":
                return kl_weight.sum() + kl_bias.sum()
            return kl_weight.mean() + kl_bias.mean()
        raise AttributeError("Posterior not set. kl() needs to be called after a forward pass.")

    def forward(self, x):
        if self._use_cached_posterior and hasattr(self, "_cached_weight") and hasattr(self, "_cached_bias"):
            weight = self._cached_weight
            bias = self._cached_bias
        else:
            #
            self.weight_posterior = Normal(self.weight_mean, self.weight_var.sqrt())
            weight = self.weight_posterior.rsample()
            self._cached_weight = weight
            #
            self.bias_posterior = Normal(self.bias_mean, self.bias_var.sqrt())
            bias = self.bias_posterior.rsample()
            self._cached_bias = bias
            #
        self._use_cached_posterior = False  # Reset this, so it has to be called explicitly every time.
        return F.linear(x, weight, bias)
