import torch

from .quantizer import Quantizer


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


def clip(x, eps):
    x_clip = torch.where(x > eps, x, eps)
    return x - x.detach() + x_clip.detach()


def round_p2(x):
    y = torch.log2(x).round()
    y = 2 ** y
    y_grad = x
    return (y - y_grad).detach() + y_grad


class LsqQuantizer(Quantizer):
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
        init=False,
        apot=False,
        out_channels=None,
        **kwargs,
    ):
        super().__init__(
            bit, thd_pos, thd_neg, all_positive, symmetric, per_channel, normalize_first
        )
        self.per_channel = per_channel
        if self.per_channel:
            assert out_channels is not None
            self.scale = torch.nn.Parameter(torch.ones(out_channels,1))
        else:
            self.scale = torch.nn.Parameter(torch.ones(1))
        # Whether use additive power of 2
        self.apot = apot
        if apot:
            if self.per_channel:
                self.a = torch.nn.Parameter(torch.ones(out_channels,1))
            else:
                self.a = torch.nn.Parameter(torch.ones(1)) # scale = a * scale where a is integer and scale is power of 2
        else:
            self.a = torch.tensor(1.0)
        self.p2_round_scale = p2_round_scale
        self.alpha = 0.0
        self.init = init
        self.value_int = 0

    def init_from(self, x):
        x = self.normalize(x)
        if self.per_channel:
            self.scale.data.copy_(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True).reshape(-1, 1) * 2 / (self.thd_pos ** 0.5))
        else:
            self.scale.data.copy_(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))
        self.init = True

    def quantize(self, x):
        if not self.init:
            self.init_from(x)
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x[0].numel()) ** 0.5)
            # s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(clip(self.scale, torch.tensor(self.eps).float().to(self.scale.device)), s_grad_scale)
        # print("self.eps",self.eps)
        # s_scale = grad_scale(self.s, s_grad_scale)
        a_scale = grad_scale(self.a, s_grad_scale)
        # round to power-of-2
        if self.p2_round_scale:
            s_scale = round_p2(s_scale)
            a_scale = round_pass(a_scale)
            s_scale = s_scale * a_scale
        if self.per_channel:
            x = x / s_scale.repeat(1, x[0].numel()).reshape(x.shape)
        else:
            x = x / s_scale
        if self.bit == 1 and not self.all_positive:
            x = torch.sign(x)
        else:
            x = torch.clamp(x, self.thd_neg, self.thd_pos)
            x = round_pass(x)
        self.alpha = s_scale
        return x
    
    def detach_quantize_weight(self, x):
        if self.per_channel:
            tmp_s_grad_scale = 1.0 / ((self.thd_pos * x[0].numel()) ** 0.5)
        else:
            tmp_s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        tmp_scale = self.scale.detach().data
        tmp_s_scale = clip(tmp_scale, torch.tensor(self.eps).float().to(self.scale.device)) * tmp_s_grad_scale

        if self.p2_round_scale:
            tmp_s_scale = 2 ** (torch.log2(tmp_s_scale).round())
            # tmp_s_scale = torch.ones_like(tmp_scale)

        if self.per_channel:
            x_quantized = x / tmp_s_scale.repeat(1, x[0].numel()).reshape(x.shape)
        else:
            x_quantized = x / tmp_s_scale
        
        if self.bit == 1 and not self.all_positive:
            x_quantized = torch.sign(x_quantized)
        else:
            x_quantized = torch.clamp(x_quantized, self.thd_neg, self.thd_pos)
            x_quantized = round_pass(x_quantized)
        
        # 返回计算结果，但不修改 self
        return x_quantized
    
    def dequantize(self, x):
        s_scale = self.alpha
        if self.per_channel:
            x = x * s_scale.repeat(1, x[0].numel()).reshape(x.shape)
        else:
            x = x * s_scale
        return x
    
    def forward(self, x):
        x = self.quantize(x)
        # if self.per_channel:
        #     self.value_int = x
        x = self.dequantize(x)
        return x

    def set_bw(self, bit):
        self.init=False
        self.bit = bit
        if self.all_positive:
            if bit == 1:
                self.thd_neg = 0
                self.thd_pos = 1
            elif self.symmetric:
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
            elif self.symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1
    
    def extra_repr(self):
        return (
            f"bit={self.bit}, "
            f"pos={self.thd_pos}, neg={self.thd_neg}, "
            f"norm=({self.normalize_first}, {self.eps}, {self.gamma}), "
            f"all_positive={self.all_positive}, "
            f"symmetric={self.symmetric}, "
            f"per_channel={self.per_channel}, "
            f"apot={self.apot} "
        )
