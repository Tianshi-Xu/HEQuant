import torch
import torch.nn as nn

from .quantizer import Quantizer


class SymQuantizer(torch.autograd.Function):
    """
        uniform quantization
    """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, p2_round_scale):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        ctx.num_bits = num_bits
        if num_bits < 32:
            input = torch.where(input < clip_val[1], input, clip_val[1])
            input = torch.where(input > clip_val[0], input, clip_val[0])
            # print(clip_val)
            # print(num_bits)
            # NOTE: dynamic scaling (max_input).
            if layerwise:
                max_input = torch.max(torch.abs(input)).expand_as(input)
            else:
                if input.ndimension() <= 3:
                    # weight & hidden layer
                    max_input = torch.max(torch.abs(input), dim=-1, keepdim=True)[0].expand_as(input).detach()
                elif input.ndimension() == 4:
                    # TODO: attention score matrix, calculate alpha / beta per head
                    
                    # print("input shape",input.shape)
                    tmp = input.view(input.shape[0], input.shape[1], -1)
                    # print("tmp shape",tmp.shape)
                    max_input = torch.max(torch.abs(tmp), dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(
                        input).detach()
                    # print("max_input.shape",max_input.shape)
                    # print(torch.unique(max_input))
                else:
                    raise ValueError
            s = (2 ** (num_bits - 1) - 1) / max_input
            # if p2_round_scale:
            #     s = 2 ** (torch.log2(s).round())
            output = torch.round(input * s).div(s)
        else:
            output = input

        return output,s

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        num_bits = ctx.num_bits
        grad_input = grad_output.clone()
        grad_clip = None
        if num_bits < 32:
            grad_input[input.ge(clip_val[1])] = 0
            grad_input[input.le(clip_val[0])] = 0
            # refer to PACT
            grad_clip_pos = (grad_output * input.ge(clip_val[1]).float()).sum()
            grad_clip_neg = (grad_output * (input.le(clip_val[0]).float())).sum()
            grad_clip = torch.tensor([grad_clip_neg, grad_clip_pos]).to(input.device)
        return grad_input, grad_clip, None, None, None


class PACTQuantizer(Quantizer):
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
        init_val = 2.5
        self.register_buffer('clip_val', torch.tensor([-init_val, init_val]))
        self.clip_val = nn.Parameter(self.clip_val)
        self.p2_round_scale = p2_round_scale
        self.alpha=1.0

    def forward(self, x):
        quant_fn = SymQuantizer.apply
        x_q,alpha= quant_fn(
            x, self.clip_val, self.bit, not self.per_channel, self.p2_round_scale
        )
        self.alpha=alpha
        return x_q

    def extra_repr(self):
        return (
            f"bit={self.bit}, pose={self.thd_pos}, neg={self.thd_neg}, "
            "norm=({self.normalize_first}, "
            f"{self.eps}, {self.gamma}), "
            f"all_positive={self.all_positive}, "
            f"symmetric={self.symmetric}, "
            f"per_channel={self.per_channel}"
        )
