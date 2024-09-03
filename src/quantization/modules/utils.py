import os
import torch
import torch.nn as nn
from .conv import QConv2d, QConvBn2d
from .linear import QLinear
from .attention import QAttention
from ..quantizer import build_quantizer
from torch.ao.nn.intrinsic import ConvBn2d
from src.utils import Attention
import math
import matplotlib.pyplot as plt
import numpy as np

QMODULE_MAPPINGS = {
    torch.nn.Conv2d: QConv2d,
    torch.nn.Linear: QLinear,
    Attention: QAttention,
    ConvBn2d: QConvBn2d,
    # nn.Sequential:QConvBn2d,
    # torch.nn.intrinsic.modules.fused.ConvBn2d: QConvBn2d,
}


def get_module_by_name(model, module_name):
    names = module_name.split(".")
    module = model
    for name in names:
        module = getattr(module, name)
    return module


def set_module_by_name(model, module_name, module):
    names = module_name.split(".")
    if len(names) == 1:
        parent = model
    else:
        parent = get_module_by_name(model, ".".join(names[:-1]))
    setattr(parent, names[-1], module)


def replace_module_by_qmodule(model, qconfigs):
    for name, cfg in qconfigs.items():
        module = get_module_by_name(model, name)
        if isinstance(module, ConvBn2d):
            wq = {"out_channels": module[0].out_channels}
        elif isinstance(module, nn.Conv2d):
            wq = {"out_channels": module.out_channels}
        elif isinstance(module, nn.Linear):
            wq = {"out_channels": module.out_features}
        cfg['weight'].update(wq)
        cfg['act'].update(wq)
        # print(cfg['weight'])
        qmodule = QMODULE_MAPPINGS[type(module)](
            module,
            quan_w_fn=build_quantizer(cfg["weight"]),
            quan_a_fn=build_quantizer(cfg["act"]),
            # quan_attn_fn=build_quantizer(getattr(cfg, "attn", cfg["act"])),
        )
        set_module_by_name(model, name, qmodule)
    return model


def register_act_quant_hook(model, qconfigs):

    def quant_act_hook(self, input, output):
        # return self.quan_a_fn(output)
        # max = torch.max(abs(output))
        # f = open("/home/rjwei/project/new-Ternary-ViT/magnitude_a=0.226.txt","a")
        # f.write(f'{max.item()} \n')
        # f.close()

        # global i
        # if i < 19:
        #     i = i+1
        # else:
        #     i = 0
        # print(i)
        # scale = var_list[i] / 5.41e-2

        # f = open("/home/rjwei/project/new-Ternary-ViT/3.txt","a")
        # f.write(f'{var.item()} \n')
        # f.close()
 
        return self.quan_a_fn(output) 

    for name, cfg in qconfigs.items():
        if cfg is not None:
            module = get_module_by_name(model, name) #根据name找到module
            quan_a_fn = build_quantizer(cfg.get("act", cfg))
            module.quan_a_fn = quan_a_fn
            module.register_forward_hook(quant_act_hook) #对每个module使用hook函数
    return model


def register_before_relu_quant_hook(model, qconfigs):

    def quant_act_hook(self, input):
        input = input[0]
        return self.quan_a_fn(input)

    for name, cfg in qconfigs.items():  
        if cfg is not None:
            module = get_module_by_name(model, name)
            quan_a_fn = build_quantizer(cfg.get("act", cfg))
            module.quan_a_fn = quan_a_fn
            module.register_forward_pre_hook(quant_act_hook) 
    return model


# Error injection for Huyixuan CIM
# BN前考虑conv+residual , 应当按这个注入误差

# global i
# i = 0
def register_error_hook(model, econfigs):

    def error_inject_hook(self, input):

        input = input[0]
        std = torch.std(input)
        # print(std, flush=True)
        # std = torch.std(output)

        # '''plot conv output + residual (i.e. BN input) distribution'''
        # input = input / cfg
        # input_plot = input.reshape(-1).cpu().detach().numpy()
        # plt.clf()
        # plt.hist(input_plot, bins=100)

        # global i
        # plt.savefig(f'/home/rjwei/project/new-Ternary-ViT/fig/conv_output_div_alpha/{i}.png')
        # i = i + 1
        

        #return input
        return input + std * cfg * torch.randn_like(input)
    


    for name, cfg in econfigs.items():
        if cfg is not None:
            module = get_module_by_name(model, name) #根据name找到module
            module.register_forward_pre_hook(error_inject_hook) #对每个module使用hook函数
            #module.register_forward_hook(error_inject_hook) #对每个module使用hook函数

    return model


# '''Error injection for Huyixuan CIM
#     conv output
# '''

# # global j
# # j = 0
# def register_error_hook(model, econfigs):

#     def error_inject_hook(self, input, output):


#         std = torch.std(output)

#         '''plot conv output distribution'''
#         output = output / cfg
#         output_plot = output.reshape(-1).cpu().detach().numpy()

#         # plt.clf()
#         # plt.hist(output_plot, bins=100)

#         # global j
#         # plt.savefig(f'/home/rjwei/project/new-Ternary-ViT/fig/conv_output_div_alpha/{j}.png')
#         # j = j + 1

#         # global j
#         # with open(f'conv_output{j}.txt', 'wb') as f: 
#         #     for k in output_plot:
#         #         f.write(str(k).encode())
#         #         f.write('\n'.encode())
#         # j = j + 1

#         return output
#         #return output + std * cfg * torch.randn_like(output)


#     for name, cfg in econfigs.items():
#         if cfg is not None:
#             module = get_module_by_name(model, name) #根据name找到module
#             module.register_forward_hook(error_inject_hook) #对每个module使用hook函数

#     return model


# Error injection for QiaoXin CIM
# def register_act_quant_hook(model, qconfigs):

#     def quant_act_hook(self, input, output):

#         chns = output.size(1)
#         scale = math.sqrt(math.ceil(chns / 256) * 3 * 1.722e-4)

#         return self.quan_a_fn(output) + scale * torch.randn_like(output)
#         #return self.quan_a_fn(output)

#     for name, cfg in qconfigs.items():
#         if cfg is not None:
#             module = get_module_by_name(model, name)
#             quan_a_fn = build_quantizer(cfg.get("act", cfg))
#             module.quan_a_fn = quan_a_fn
#             module.register_forward_hook(quant_act_hook)

#     return model
