import torch


class Quantizer(torch.nn.Module):
    def __init__(
        self,
        bit=None,
        thd_pos=None,
        thd_neg=None,
        all_positive=False,
        symmetric=False,
        per_channel=False,
        normalize_first=False,
        **kwargs,
    ):
        super().__init__()

        assert not (bit is None and thd_pos is None and thd_neg is None)
        if bit is None:
            self.thd_neg = thd_neg
            self.thd_pos = thd_pos
        else:
            if all_positive:
                if bit == 1:
                    self.thd_neg = 0
                    self.thd_pos = 1
                elif symmetric:
                    # unsigned activation is quantized to [0, 2^b-2]
                    self.thd_neg = 0
                    self.thd_pos = 2 ** bit - 2
                else:
                    # unsigned activation is quantized to [0, 2^b-1]
                    self.thd_neg = 0
                    self.thd_pos = 2 ** bit - 1
            else:
                if bit == 1:
                    self.thd_neg = -1
                    self.thd_pos = 1
                elif symmetric:
                    # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                    self.thd_neg = - 2 ** (bit - 1) + 1
                    self.thd_pos = 2 ** (bit - 1) - 1
                else:
                    # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                    self.thd_neg = - 2 ** (bit - 1)
                    self.thd_pos = 2 ** (bit - 1) - 1

        self.eps = 1e-5
        self.all_positive = all_positive
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.gamma = 1.0
        self.bit = bit
        self.normalize_first = normalize_first

    def normalize(self, x):
        if self.normalize_first:
            std, mean = torch.std_mean(x, dim=list(range(1, x.ndim)),
                    keepdim=True, unbiased=False)
            scale = self.gamma * x[0].numel() ** -0.5
            return scale * (x - mean) / (std + self.eps)
        else:
            return x

    def init_from(self, x, *args, **kwargs):
        pass

    def forward(self, x):
        raise NotImplementedError

    def extra_repr(self):
        return (
            f"bit={self.bit}, norm=({self.normalize_first}, "
            f"{self.eps}, {self.gamma})"
        )


class IdentityQuantizer(Quantizer):
    def __init__(self, normalize_first=False, *args, **kwargs):
        super().__init__(bit=32, normalize_first=normalize_first)

    def forward(self, x):
        x = self.normalize(x)
        return x
