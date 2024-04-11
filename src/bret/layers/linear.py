import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianLinear(nn.Module):
    def __init__(self, prior, prior_var=1.0):
        super().__init__()
        self.weight_prior_mean = torch.tensor(prior.weight.data, dtype=prior.weight.dtype, device=prior.weight.device)
        self.weight_prior_var = prior_var
        self.weight_mean = prior.weight
        self.weight_logvar = nn.Parameter(-torch.rand_like(prior.weight))
        self.bias = prior.bias
        if self.bias is not None:
            # We don't have to sample/train the bias, as it's just an offset to the mean.
            self.bias.requires_grad = False

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
        num_tokens = x.size(1)
        out_features_dim = self.weight_mean.size(0)
        in_features_dim = self.weight_mean.size(1)
        z = (
            torch.randn(
                (batch_size, *self.weight_mean.shape), dtype=self.weight_mean.dtype, device=self.weight_mean.device
            )
            * self.weight_var.sqrt()
        )
        out = F.linear(
            x,
            (self.weight_mean + z).reshape(batch_size * out_features_dim, in_features_dim),
            self.bias.repeat(batch_size),
        )
        out = out.reshape(batch_size, num_tokens, batch_size, out_features_dim).mean(2)
        return out
