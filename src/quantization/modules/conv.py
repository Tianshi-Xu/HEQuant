import copy
import torch


class QConv2d(torch.nn.Conv2d):
    def __init__(self, m: torch.nn.Conv2d, quan_w_fn=None, quan_a_fn=None, **kwargs):
        assert type(m) == torch.nn.Conv2d
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn

        self.weight = torch.nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = torch.nn.Parameter(m.bias.detach())

    def forward(self, x):
        # print("before quant, x: ",torch.min(x),torch.max(x))
        # print("before quant, w: ",self.weight.min(),self.weight.max())
        quantized_weight = self.quan_w_fn(self.weight)
        quantized_act = self.quan_a_fn(x)
        # print(self.quan_w_fn)
        # print(self.quan_a_fn)
        # print("after quant, x: ",torch.min(quantized_act),torch.max(quantized_act))
        # print("after quant, w: ",torch.min(quantized_weight),torch.max(quantized_weight))
        # print(quantized_act)
        # print("----------")
        # print(torch.unique(quantized_weight[0,:,:,:]))
        return self._conv_forward(quantized_act, quantized_weight, self.bias)


class QConvBn2d(torch.nn.Conv2d):
    def __init__(
        self, m, quan_w_fn=None, quan_a_fn=None
    ):
        assert type(m) == torch.nn.intrinsic.modules.fused.ConvBn2d
        conv, bn = m[0], m[1]
        super().__init__(conv.in_channels, conv.out_channels, conv.kernel_size,
                         stride=conv.stride,
                         padding=conv.padding,
                         dilation=conv.dilation,
                         groups=conv.groups,
                         bias=True if conv.bias is not None else False,
                         padding_mode=conv.padding_mode)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn

        self.weight = torch.nn.Parameter(conv.weight.detach())
        if conv.bias is not None:
            self.bias = torch.nn.Parameter(conv.bias.detach())
        self.bn = copy.deepcopy(bn)

        scale_factor, _, _ = self._bn_scaling_factor()
        self.quan_w_fn.init_from(conv.weight * scale_factor)
        
        self.d1 = None

    def _bn_scaling_factor(self):
        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        weight_shape = [1] * len(self.weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(self.weight.shape) 
        bias_shape[1] = -1
        scale_factor = scale_factor.reshape(weight_shape)
        return scale_factor, weight_shape, bias_shape

    def forward(self, x):
        if self.d1 is None:
            assert x.shape[2] == x.shape[3]
            self.d1 = x.shape[2]*x.shape[3]
        scale_factor, _, bias_shape = self._bn_scaling_factor()
        scaled_weight = self.quan_w_fn(self.weight * scale_factor)
        if self.bias is not None:
            zero_bias = torch.zeros_like(self.bias)
        else:
            zero_bias = torch.zeros(self.out_channels, device=scaled_weight.device)
        y = self._conv_forward(self.quan_a_fn(x), scaled_weight, zero_bias)
        y_orig = y / scale_factor.reshape(bias_shape)
        if self.bias is not None:
            y_orig = y_orig + self.bias.reshape(bias_shape)
        y = self.bn(y_orig)
        return y
