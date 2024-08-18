import torch

from .quantizer import Quantizer


class hbcQuantizer(Quantizer):
    def __init__(
        self,
        bit=None,
        thd_pos=None,
        thd_neg=None,
        all_positive=False,
        symmetric=False,
        per_channel=True,
        normalize_first=False,
        **kwargs,
    ):
        super().__init__(
            bit, thd_pos, thd_neg, all_positive, symmetric, per_channel, normalize_first
        )

        self.clipping_weight = {
            1: 1, 
            2: 1.5365634467582905,
            4: 3.125708912607169,
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
        s = clamp.detach() # s = abs_mean
        if self.bit == 1 and not self.all_positive:
            xq = torch.sign(xq)
        else:
            xq = torch.clamp(xq, self.thd_neg, self.thd_pos)
        xq = (xq * s).permute(1, 0).reshape(orig_shape)
        cliped_x = torch.clamp(x, -1.0, 1.0)
        return (xq - cliped_x).detach() + cliped_x

    def extra_repr(self):
        return (
            f"bit={self.bit}, pose={self.thd_pos}, neg={self.thd_neg}, "
            "norm=({self.normalize_first}, "
            f"{self.eps}, {self.gamma}), "
            f"all_positive={self.all_positive}, "
            f"symmetric={self.symmetric}, "
            f"per_channel={self.per_channel}"
        )
