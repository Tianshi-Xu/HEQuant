import torch
import torch.nn as nn
import logging
import math

from .quantizer import Quantizer


def round_p2(x):
    y = torch.log2(x).round()
    y = 2 ** y
    y_grad = x
    return (y - y_grad).detach() + y_grad

class LsqStepSize(nn.Parameter):
    def __init__(self, tensor):
        super(LsqStepSize, self).__new__(nn.Parameter, data=tensor)
        self.initialized = False

    def _initialize(self, init_tensor):
        assert not self.initialized, 'already initialized.'
        self.data.copy_(init_tensor)
        # print('Stepsize initialized to %.6f' % self.data.item())
        self.initialized = True

    def initialize_wrapper(self, tensor, num_bits, symmetric, init_method='default'):
        # input: everthing needed to initialize step_size
        Qp = 2 ** (num_bits - 1) - 1 if symmetric else 2 ** (num_bits) - 1
        if init_method == 'default':
            init_val = 2 * tensor.abs().mean() / math.sqrt(Qp) if symmetric \
                else 4 * tensor.abs().mean() / math.sqrt(Qp)
        elif init_method == 'uniform':
            init_val = 1./(2*Qp+1) if symmetric else 1./Qp

        self._initialize(init_val)


class AsymLsqQuantizer(torch.autograd.Function):
    """
        Asymetric LSQ quantization. Modified from LSQ.
    """
    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise, p2_round_scale):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        if not layerwise:
            # TODO
            raise NotImplementedError
        ctx.num_bits = num_bits
        if num_bits == 32:
            return input

        Qn = 0
        Qp = 2 ** (num_bits) - 1
        # asymmetric: make sure input \in [0, +\inf], remember to add it back
        min_val = input.min().item()
        input_ = input - min_val

        if not alpha.initialized:# alpha.item() == 1.0 and (not alpha.initialized):
            alpha.initialize_wrapper(input, num_bits, symmetric=False, init_method='default')
        # print(alpha)
        assert alpha > 0#, 'alpha = {:.6f} becomes non-positive'.format(alpha)

        grad_scale = 1.0 / math.sqrt(input.numel() * Qp)
        # grad_scale = 1.0
        ctx.save_for_backward(input_, alpha)
        ctx.other = grad_scale, Qn, Qp
        if p2_round_scale:
            alpha = round_p2(alpha)
        q_w = (input_ / alpha).round().clamp(Qn, Qp)
        # print("alpha")
        # print(alpha)
        w_q = q_w * alpha
        w_q = w_q + min_val
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits == 32:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big   # this is more cpu-friendly than torch.ones(input_.shape)
        grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                -q_w + q_w.round())) * grad_output * grad_scale).sum().unsqueeze(dim=0)
        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None, None

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
        **kwargs,
    ):
        super().__init__(
            bit, thd_pos, thd_neg, all_positive, symmetric, per_channel, normalize_first
        )
        self.clip_val = LsqStepSize(torch.tensor(1.0))
        self.p2_round_scale = p2_round_scale
        self.alpha=1.0
        # self.eps = self.register_buffer("eps", torch.tensor(1e-5).float())

    def forward(self, x):
        quant_fn = AsymLsqQuantizer.apply
        # print(quant_fn)
        return quant_fn(
            x, self.clip_val, self.bit, not self.per_channel, self.p2_round_scale
        )

    def extra_repr(self):
        return (
            f"bit={self.bit}, "
            f"pos={self.thd_pos}, neg={self.thd_neg}, "
            f"norm=({self.normalize_first}, {self.eps}, {self.gamma}), "
            f"all_positive={self.all_positive}, "
            f"symmetric=False, "
            f"per_channel={self.per_channel}"
        )
