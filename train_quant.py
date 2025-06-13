#!/usr/bin/env python3
# This is a slightly modified version of timm's training script
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from logging.handlers import RotatingFileHandler
from src.utils.hess.myhessian import hessian # Hessian computation
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils
import math
import util.misc as misc

from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, \
     model_parameters
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler, setup_default_logging, ModelEmaV2, get_outdir, distribute_bn, CheckpointSaver, update_summary,AverageMeter,dispatch_clip_grad, accuracy, reduce_tensor
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from src.quantization.modules.conv import QConv2d, QConvBn2d
from src import *
from src.quantization.utils import KLLossSoft
from util.datasets import build_dataset
from util.utils import train_one_epoch, validate

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass


torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset / Model parameters
parser.add_argument('data_dir', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')
parser.add_argument('--model', default='resnet101', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N',
                    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
                    help='ratio of validation batch size to training batch size (default: 1)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0001,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')

# Learning rate schedule parameters
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=None, metavar='PCT',
                    help='Color jitter factor (default: None)')
parser.add_argument('--aa', type=str, default="rand-m9-mstd0.5-inc1", metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "const")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Misc
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--log-wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')
# Quant
parser.add_argument('--wq-enable', action='store_true', default=False,
                    help='enable weight quantization')
parser.add_argument('--wq-mode', default='LSQ', type=str,
                    help='weight quantization mode')
parser.add_argument('--wq-bitw', default=None, type=int,
                    help='weight quantization bit width')
parser.add_argument('--wq-pos', default=None, type=int,
                    help='weight quantization positive threshold')
parser.add_argument('--wq-neg', default=None, type=int,
                    help='weight quantization negative threshold')
parser.add_argument('--wq-per-channel', default=False, action='store_true',
                    help='per channel weight quantization')
parser.add_argument('--wq-asym', action='store_true', default=False,
                    help='asymmetric quantization for weight')
parser.add_argument('--aq-enable', action='store_true', default=False,
                    help='enable act quantization')
parser.add_argument('--aq-mode', default='lsq', type=str,
                    help='act quantization mode')
parser.add_argument('--aq-bitw', default=None, type=int,
                    help='act quantization bit width')
parser.add_argument('--aq-pos', default=None, type=int,
                    help='act quantization positive threshold')
parser.add_argument('--aq-neg', default=None, type=int,
                    help='act quantization negative threshold')
parser.add_argument('--aq-asym', action='store_true', default=False,
                    help='asymmetric quantization for activation')
parser.add_argument('--qmodules', type=str, nargs='+', default=None, metavar='N',
                    help='Quantized modules')
parser.add_argument('--resq-modules', type=str, nargs='+', default=None, metavar='N',
                    help='Quantized residual')
parser.add_argument('--resq-enable', action='store_true', default=False,
                    help='enable residual quantization')
parser.add_argument('--resq-mode', default='lsq', type=str,
                    help='residual quantization mode')
parser.add_argument('--resq-bitw', default=None, type=int,
                    help='residual quantization bit width')
parser.add_argument('--resq-pos', default=None, type=int,
                    help='residual quantization positive threshold')
parser.add_argument('--resq-neg', default=None, type=int,
                    help='residual quantization negative threshold')
parser.add_argument('--resq-asym', action='store_true', default=False,
                    help='asymmetric quantization for residual')
parser.add_argument('--aq_per_channel', action='store_true', default=False,
                    help='per-channel quantization for activation')
parser.add_argument('--wq_asym', action='store_true', default=False,
                    help='assymentric quantization for weight')
parser.add_argument('--aq_asym', action='store_true', default=True,
                    help='assymentric quantization for activation')
parser.add_argument('--powerof2', action='store_true', default=True,
                    help='whether scale is power of 2')
# distributed train
parser.add_argument(
    "--world_size", default=1, type=int, help="number of distributed processes"
)
parser.add_argument("--local-rank", default=-1, type=int)
parser.add_argument("--dist_on_itp", action="store_true")
parser.add_argument(
    "--dist_url", default="env://", help="url used to set up distributed training"
)
parser.add_argument(
    "--device", default="cuda", help="device to use for training / testing"
)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument(
    "--dist_eval",
    action="store_true",
    default=False,
    help="Enabling distributed evaluation (recommended during training for faster monitor",
)
# KD
parser.add_argument('--use-kd', action='store_true', default=False,
                    help='whether to use kd')
parser.add_argument('--kd-alpha', default=1.0, type=float,
                    help='KD alpha, soft loss portion (default: 1.0)')
parser.add_argument('--teacher', default='resnet101', type=str, metavar='MODEL',
                    help='Name of teacher model (default: "countception"')
parser.add_argument('--teacher-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize teacher model from this checkpoint (default: none)')
# Quant_ILP
parser.add_argument('--log_name', default='none', type=str,
                    help='act sparsification pattern')
parser.add_argument('--budget', default=1,
                    help='budget, latency ratio, 4 means w4a4, and the like')
parser.add_argument('--bw_list', default="", type=str,
                    help='loool')
parser.add_argument('--ba_list', default="", type=str,
                    help='loool')
parser.add_argument('--reg_weight', default=1e-3, type=float,
                    help='loool')
lasso_beta=0
origin_latency = 0
rotate_mats = {}
rev_rotate_mats = {}

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def next_power_2(d):
    p = math.ceil(math.log2(d))
    return int(pow(2,p))

def create_teacher_model(args):
    teacher = create_model(
        args.teacher,
        pretrained=False,
        num_classes=args.num_classes,
        drop_rate=0.,
        drop_connect_rate=0.,
        drop_block_rate=0.,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        # checkpoint_path=args.teacher_checkpoint,
        # checkpoint_path="",
    )
    # if args.teacher_checkpoint != "":
        # tmp = torch.load(args.teacher_checkpoint,weights_only=False)
        # print(tmp.keys())
        # load_checkpoint(teacher, args.teacher_checkpoint, strict=True)
    teacher = teacher.eval()
    return teacher


def get_qat_model(model, args):
    
    def _decode_args(m):
        if ";" not in m:
            return m, {}, {}
        args = m.split(";")
        name = args[0]
        ret = {"wq": {}, "aq": {}}
        for arg in args[1:]:
            # print(arg)
            val = arg.split(":")
            assert val[1] not in ret[val[0]]
            ret[val[0]][val[1]] = eval(val[2])
        return name, ret["wq"], ret["aq"]
    
    qat_model = copy.deepcopy(model)
    qat_model.train()

    qconfigs = {}
    if args.qmodules is not None:
        for m in args.qmodules:
            mod, wq, aq = _decode_args(m)
            wcfg = {
                "enable": args.wq_enable,
                "mode": args.wq_mode if args.wq_enable else "Identity",
                "bit": args.wq_bitw,
                "thd_pos": args.wq_pos,
                "thd_neg": args.wq_neg,
                "all_positive": False,
                "symmetric": not args.wq_asym,  # not args.wq_asym,
                "per_channel": args.wq_per_channel,
                "normalize_first": False,
                "p2_round_scale": args.powerof2,  # scaling factor is power-of-2?
                "apot": False,
            }
            wcfg.update(wq)
            acfg = {
                "enable": args.aq_enable,
                "mode": args.aq_mode if args.aq_enable else "Identity",
                "bit": args.aq_bitw,
                "thd_pos": args.aq_pos,
                "thd_neg": args.aq_neg,
                "all_positive": False if "convbn_first" in m else args.aq_asym,
                "symmetric": True if "convbn_first" in m else not args.aq_asym,  # not args.aq_asym,
                "per_channel": args.aq_per_channel,
                "normalize_first": False,
                "p2_round_scale": args.powerof2,  # scaling factor is power-of-2?
            }
            acfg.update(aq)
            qconfigs[mod] = {"weight": wcfg, "act": acfg}
        qat_model = replace_module_by_qmodule(model, qconfigs)
    if args.resq_modules is not None:
        # quantize residual 
        qconfigs = {}
        for m in args.resq_modules:
            acfg = {
                "enable": args.resq_enable,
                "mode": args.resq_mode if args.resq_enable else "Identity",
                "bit": args.resq_bitw,
                "thd_pos": args.resq_pos,
                "thd_neg": args.resq_neg,
                # "all_positive": args.use_relu,
                "all_positive": False if "downsample" in m else True,
                "symmetric": True if "downsample" in m else False,  # not args.resq_asym,
                "per_channel": False,
                "normalize_first": False,
                "p2_round_scale": True,  # scaling factor is power-of-2?
            }
            qconfigs[m] = {"act": acfg}
        
        qat_model = register_act_quant_hook(model, qconfigs)

    return qat_model
    
def get_qat_fp_model(model, args):
    
    def _decode_args(m):
        if ";" not in m:
            return m, {}, {}
        args = m.split(";")
        name = args[0]
        ret = {"wq": {}, "aq": {}}
        for arg in args[1:]:
            # print(arg)
            val = arg.split(":")
            assert val[1] not in ret[val[0]]
            ret[val[0]][val[1]] = eval(val[2])
        return name, ret["wq"], ret["aq"]
    
    qat_model = copy.deepcopy(model)
    qat_model.train()

    qconfigs = {}
    if args.qmodules is not None:
        for m in args.qmodules:
            mod, wq, aq = _decode_args(m)
            wcfg = {
                "enable": args.wq_enable,
                "mode": "Identity",
                "bit": 32,
                "thd_pos": args.wq_pos,
                "thd_neg": args.wq_neg,
                "all_positive": False,
                "symmetric": not args.wq_asym,  # not args.wq_asym,
                "per_channel": args.wq_per_channel,
                "normalize_first": False,
                "p2_round_scale": args.powerof2,  # scaling factor is power-of-2?
                "apot": False,
            }
            wcfg.update(wq)
            acfg = {
                "enable": args.aq_enable,
                "mode": "Identity",
                "bit": 32,
                "thd_pos": args.aq_pos,
                "thd_neg": args.aq_neg,
                "all_positive": False if "convbn_first" in m else args.aq_asym,
                "symmetric": True if "convbn_first" in m else not args.aq_asym,  # not args.aq_asym,
                "per_channel": args.aq_per_channel,
                "normalize_first": False,
                "p2_round_scale": args.powerof2,  # scaling factor is power-of-2?
            }
            acfg.update(aq)
            qconfigs[mod] = {"weight": wcfg, "act": acfg}
        qat_model = replace_module_by_qmodule(model, qconfigs)
    if args.resq_modules is not None:
        # quantize residual 
        qconfigs = {}
        for m in args.resq_modules:
            acfg = {
                "enable": args.resq_enable,
                "mode": "Identity",
                "bit": 32,
                "thd_pos": args.resq_pos,
                "thd_neg": args.resq_neg,
                # "all_positive": args.use_relu,
                "all_positive": False if "downsample" in m else True,
                "symmetric": True if "downsample" in m else False,  # not args.resq_asym,
                "per_channel": False,
                "normalize_first": False,
                "p2_round_scale": True,  # scaling factor is power-of-2?
            }
            qconfigs[m] = {"act": acfg}
        
        qat_model = register_act_quant_hook(model, qconfigs)

    return qat_model
    

def main():
    args, args_text = _parse_args()
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()
    device = torch.device(args.device)
    print("global_rank", global_rank)
    if global_rank == 0:
        setup_default_logging()
        handler = RotatingFileHandler(args.log_name+'.log', maxBytes=10*1024*1024, backupCount=5)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        _logger.addHandler(handler)
        _logger.info(args)
    args.local_rank = global_rank
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # build dataset
    dataset_train = build_dataset(is_train=True, args=args)
    dataset_eval = build_dataset(is_train=False, args=args)
    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_eval) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_eval, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_eval)
            
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_eval,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes,
        )
        
    # build model
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript)
    model.to(device)
    if mixup_active:
        # smoothing is handled with mixup target transform
        train_loss_fn = SoftTargetCrossEntropy().cuda(device)
    elif args.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda(device)
    else:
        train_loss_fn = nn.CrossEntropyLoss().cuda(device)
    train_loss_fn_kd = None
    if args.use_kd:    
        train_loss_fn_kd = KLLossSoft().cuda(device)
    validate_loss_fn = nn.CrossEntropyLoss().cuda(device)
    

        # if global_rank == 0:
        #     _logger.info("Verifying initial model in test dataset first time")
        # validate(model, data_loader_val, validate_loss_fn, args, _logger)
    
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if global_rank == 0:
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    teacher = None
    if args.use_kd:
        teacher = create_teacher_model(args)
        teacher = get_qat_fp_model(teacher, args)
        load_checkpoint(teacher, args.teacher_checkpoint, strict=True)
        teacher.cuda()
    
    model = get_qat_model(model, args)
    model.cuda()
    if args.initial_checkpoint:
        load_checkpoint(model, args.initial_checkpoint,strict=True)
    print("args.gpu", args.gpu)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module
        if args.sync_bn:
            assert not args.split_bn
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            if global_rank == 0:
                _logger.info(
                    'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                    'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')
    else:
        model.to(device)
        if args.use_kd:
            teacher.to(device)  
        model_without_ddp = model
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
    loss_scaler = NativeScaler()
    if global_rank == 0:
        _logger.info('Using native Torch AMP. Training in mixed precision.')

    ### load resume model
    misc.load_model(args=args,model_without_ddp=model_without_ddp,optimizer=optimizer,loss_scaler=loss_scaler)
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    # num_epochs = args.epochs
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/args.epochs), last_epoch=-1)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if global_rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))
    
    if args.use_kd:
        if global_rank == 0:
            _logger.info("Verifying teacher model")
        # validate(teacher, data_loader_val, validate_loss_fn, args, _logger)
    if args.bw_list != "":
        bw_list = [int(x) for x in args.bw_list.split(',')]
        ba_list = [int(x) for x in args.ba_list.split(',')]
        idx = 0
        for name,layer in model.named_modules():
            if "first" in name:
                continue
            if isinstance(layer, QConvBn2d):
                layer.quan_w_fn.set_bw(bw_list[idx])
                layer.quan_a_fn.set_bw(ba_list[idx])
                idx += 1
            if isinstance(layer, LsqQuantizer):
                layer.init = True
    if args.initial_checkpoint != "":
        if global_rank == 0:
            _logger.info("Verifying initial model in test dataset")
            _logger.info(model.device)
        validate(model, data_loader_val, validate_loss_fn, args, _logger)
    # if global_rank == 0:
    #     _logger.info(model)
    # setup checkpoint saver and eval metric tracking
    data_config = resolve_data_config(vars(args), model=model, verbose=True)
    if args.experiment:
        exp_name = args.experiment
    else:
        exp_name = '-'.join([
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            safe_model_name(args.model),
            str(data_config['input_size'][-1])
        ])
    output_dir = None
    if global_rank == 0:
        output_dir = get_outdir(args.output if args.output else './output/train', exp_name)
        args.output_dir = output_dir
    eval_metric = args.eval_metric
    if global_rank == 0:
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)
        with open(os.path.join(output_dir, 'model.txt'), 'w') as f:
            f.write(str(model))
    max_acc = 0
    max_acc_epoch = 0
    if global_rank == 0:
        _logger.info(model)
    for epoch in range(start_epoch, num_epochs):
        if args.distributed and hasattr(data_loader_train.sampler, 'set_epoch'):
            data_loader_train.sampler.set_epoch(epoch)

        train_metrics = train_one_epoch(
            epoch, model, data_loader_train, optimizer, train_loss_fn, args, _logger,
            lr_scheduler=lr_scheduler, output_dir=output_dir,
            loss_scaler=loss_scaler, mixup_fn=mixup_fn,teacher=teacher,loss_fn_kd=train_loss_fn_kd)

        if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
            if global_rank == 0:
                _logger.info("Distributing BatchNorm running means and vars")
            distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

        eval_metrics = validate(model, data_loader_val, validate_loss_fn, args, _logger)

        if lr_scheduler is not None:
            # step LR for next epoch
            lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])
            # lr_scheduler.step()
        if global_rank == 0:
            misc.save_model(args=args,epoch=epoch,model=model,model_without_ddp=model_without_ddp,optimizer=optimizer,loss_scaler=loss_scaler,name="last.pth.tar")
            if eval_metrics[eval_metric] > max_acc:
                max_acc = eval_metrics[eval_metric]
                max_acc_epoch = epoch
                if global_rank == 0:
                    misc.save_model(args=args,epoch=epoch,model=model,model_without_ddp=model_without_ddp,optimizer=optimizer,loss_scaler=loss_scaler,name="best.pth.tar")
    if global_rank == 0:
        _logger.info("Best accuracy: {}, epoch: {}".format(max_acc, max_acc_epoch))
    

if __name__ == '__main__':
    main()
