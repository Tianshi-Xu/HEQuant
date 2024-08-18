import torch

from .quantizer import Quantizer


class TwnQuantizer(Quantizer):
    def __init__(
        self,
        bit=None,
        thd_pos=None,
        thd_neg=None,
        all_positive=False,
        symmetric=False,
        per_channel=True,
        normalize_first=False,
        p2_round_scale=False,
        **kwargs,
    ):
        super().__init__(
            bit, thd_pos, thd_neg, all_positive, symmetric, per_channel, normalize_first
        )
        self.p2_round_scale = p2_round_scale

        self.clipping_weight = {
            #1: 0.7,
            1: 1, 
            2: 1.5365634467582905,
            3: 2.325,
            4: 3.125708912607169,
            6: 5.3,
            8: 7.2
        }[bit]

    def forward(self, x):
        orig_shape = x.size()
        xq = x.reshape(orig_shape[0], -1)
        if self.per_channel:
            abs_mean = xq.abs().mean(1)
            clamp = torch.min(
                abs_mean * self.clipping_weight,
                xq.abs().max(1)[0]
            )
        else:
            abs_mean = xq.abs().mean()
            clamp = torch.min( 
                abs_mean * self.clipping_weight,
                xq.abs().max()
            )
        xq = xq.permute(1, 0)
        s = clamp / (self.thd_pos - self.thd_neg)
        s = torch.clamp(s, min=self.eps)

        if self.p2_round_scale:
            s = 2 ** (torch.log2(s).round())

        xq = xq / s
        if self.bit == 1 and not self.all_positive:
            xq = torch.sign(xq)
        else:
            xq = torch.clamp(torch.round(xq), self.thd_neg, self.thd_pos)
        xq = (xq * s).permute(1, 0).reshape(orig_shape)

        return (xq - x).detach() + x

    def extra_repr(self):
        return (
            f"bit={self.bit}, pose={self.thd_pos}, neg={self.thd_neg}, "
            "norm=({self.normalize_first}, "
            f"{self.eps}, {self.gamma}), "
            f"all_positive={self.all_positive}, "
            f"symmetric={self.symmetric}, "
            f"per_channel={self.per_channel}"
        )
