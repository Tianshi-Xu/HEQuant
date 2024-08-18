import torch
import torch.nn as nn


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(out_chn), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class QLearnableBias(nn.Module):
    def __init__(self, m, quan_w_fn=None, quan_a_fn=None, **kwargs):
        super().__init__()
        self.bias = nn.Parameter(m.bias.detach())
        self.quan_w_fn = quan_w_fn

    def forward(self, x):
        if self.quan_w_fn is not None:
            bias = self.quan_w_fn(self.bias)
        else:
            bias = self.bias
        return x + bias.expand_as(x)
