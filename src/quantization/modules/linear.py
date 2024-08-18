import torch

from .qbias import LearnableBias


class QLinear(torch.nn.Linear):
    def __init__(self, m: torch.nn.Linear, quan_w_fn=None, quan_a_fn=None, **kwargs):
        assert type(m) == torch.nn.Linear
        super().__init__(m.in_features, m.out_features,
                         bias=True if m.bias is not None else False)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn

        self.weight = torch.nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = torch.nn.Parameter(m.bias.detach())
        # self.move = LearnableBias(m.weight.shape[1])

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        # x = self.move(x)
        quantized_act = self.quan_a_fn(x)
        return torch.nn.functional.linear(quantized_act, quantized_weight, self.bias)
