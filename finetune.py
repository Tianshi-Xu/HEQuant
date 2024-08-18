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
import imp
import sys
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.utils
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as NativeDDP

import src.resnet_3stages as resnet_3stages
from src.minionn import MiniONN
from pytorchcv.model_provider import get_model as ptcv_get_model

# sys.path.append("/data/home/menglifrl/pytorch-image-models")
sys.path.append("/home/mengli/third-party/pytorch-image-models")

from timm.data import (
    create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
)
from timm.models import (
    create_model, safe_model_name, resume_checkpoint, load_checkpoint,
    convert_splitbn_model, model_parameters
)
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

from src import *
import pdb

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

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
parser.add_argument('--resume', default='minionn_a4w8.pth.tar', type=str, metavar='PATH',
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
parser.add_argument('--decay-milestones', default=[40,80],
                    help='decay milestones for MultiStep scheduler')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
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
parser.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=10, metavar='N',
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
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='const',
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
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

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

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('--checkpoint-hist', type=int, default=3, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
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
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--log-wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')
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
parser.add_argument('--fl-wq-pos', default=None, type=int,
                    help='weight quantization positive threshold')
parser.add_argument('--fl-wq-neg', default=None, type=int,
                    help='weight quantization negative threshold')
parser.add_argument('--fl-wq-per-channel', default=False, action='store_true',
                    help='per channel weight quantization')
parser.add_argument('--replace-ln-by-bn', action='store_true', default=False,
                    help='whether to use bn instead of layernorm')
parser.add_argument('--use-kd', action='store_true', default=False,
                    help='whether to use kd')
parser.add_argument('--use-token-kd', action='store_true', default=False,
                    help='whether to use token level kd')
parser.add_argument('--kd-alpha', default=1.0, type=float,
                    help='KD alpha, soft loss portion (default: 1.0)')
parser.add_argument('--teacher', default='resnet101', type=str, metavar='MODEL',
                    help='Name of teacher model (default: "countception"')
parser.add_argument('--teacher-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize teacher model from this checkpoint (default: none)')
parser.add_argument('--quant-teacher', action='store_true', default=False,
                    help='whether to quantize the teacher')
parser.add_argument('--use-distill-head', action='store_true', default=False,
                    help='whether to use two separate heads for distillation')
parser.add_argument('--use-layer-scale', action='store_true', default=False,
                    help='whether to use scaler for attention and mlp')
parser.add_argument('--use-skip', action='store_true', default=False,
                    help='whether to use skip for connection')
parser.add_argument('--use-relu', action='store_true', default=True,
                    help='whether to use relu for the model')
parser.add_argument('--no-bn', action='store_true', default=False,
                    help='whether to use bn for the model')
parser.add_argument('--kd-type', type=str, default="last",
                    help='whether to match the last or all the embeddings')
parser.add_argument('--post-res-bn', action='store_true', default=False,
                    help='whether to have bn after res')
parser.add_argument('--use-dual-skip', action='store_true', default=False,
                    help='whether to use dual skip for resnet blocks')
parser.add_argument('--down-block-type', default='default', type=str,
                    help='Downsample block type: default, avgpool, conv3x3')

parser.add_argument('--A', type=float, default=0, metavar='S',
                    help='coefficient of nomal bias to accumulation of each layer')

parser.add_argument('--bn-modules', type=str, nargs='+', default=None, metavar='N',
                    help='error inject before bn')

parser.add_argument('--conv-modules', type=str, nargs='+', default=None, metavar='N',
                    help='error inject after conv')

parser.add_argument('--quant-firstlast', action='store_true', default=False,
                    help='whether to quantize the first and the last layer')
parser.add_argument('--quant-stagewise', action='store_true', default=False,
                    help='whether to quantize stage-wise')
                
def parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)
    if args.qmodules is None:
        args.qmodules = []
    if args.resq_modules is None:
        args.resq_modules = []
    args.use_bn = not args.no_bn

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def get_qat_model(model, args):
    qat_model = copy.deepcopy(model)
    qat_model.train()

    qconfigs = {}

    i = 0
    for name, module in model.named_modules():
        if type(module)==nn.Conv2d and name in args.qmodules:
            wcfg = {
                "enable": args.wq_enable,
                "mode": args.wq_mode if args.wq_enable else "Identity",
                "bit": args.bit_setting[i],
                "thd_pos": args.wq_pos,
                "thd_neg": args.wq_neg,
                "all_positive": False,
                "symmetric": True,  # not args.wq_asym,
                "per_channel": args.wq_per_channel,
                "normalize_first": False,
                "p2_round_scale": True,  # scaling factor is power-of-2?
            }
            acfg = {
                "enable": args.aq_enable,
                "mode": args.aq_mode if args.aq_enable else "Identity",
                "bit": 4,
                "thd_pos": args.aq_pos,
                "thd_neg": args.aq_neg,
                "all_positive": True if args.use_relu and "linear" in name else False,
                "symmetric": True,  # not args.aq_asym,
                "per_channel": False,
                "normalize_first": False,
                "p2_round_scale": True,  # scaling factor is power-of-2?
            }
            qconfigs[name] = {"weight": wcfg, "act": acfg}
            i+=1

    for n in args.fl_qmodules:
        wcfg = {
            "enable": True,
            "mode": "LSQ",
            "bit": 8,
            "thd_pos": None,
            "thd_neg": None,
            "all_positive": False,
            "symmetric": True,  # not args.wq_asym,
            "per_channel": False,
            "normalize_first": False,
            "p2_round_scale": True,  # scaling factor is power-of-2?
        }
        acfg = {
            "enable": False,
            "mode": "Identity",
            "bit": 8,
            "thd_pos": None,
            "thd_neg": None,
            "all_positive": False,  # True if args.use_relu and "linear" in m else False,
            "symmetric": True,  # not args.aq_asym,
            "per_channel": False,
            "normalize_first": False,
            "p2_round_scale": True,  # scaling factor is power-of-2?
        }
        qconfigs[n] = {"weight": wcfg, "act": acfg}
    qat_model = replace_module_by_qmodule(model, qconfigs)

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
            "all_positive": False,
            "symmetric": True,  # not args.resq_asym,
            "per_channel": False,
            "normalize_first": False,
            "p2_round_scale": True,  # scaling factor is power-of-2?
        }
        qconfigs[m] = {"act": acfg}
    
    qat_model = register_act_quant_hook(model, qconfigs)

    return qat_model
    

def create_teacher_model(args):
    if 'imagenet' in args.dataset:
        teacher = create_model(
            args.teacher,
            pretrained=False,
            num_classes=args.num_classes,
            drop_rate=0.,
            drop_connect_rate=0.,
            drop_block_rate=0.,
            global_pool=args.gp,
            # bn_tf=args.bn_tf,
            bn_momentum=args.bn_momentum,
            bn_eps=args.bn_eps,
            # checkpoint_path=args.teacher_checkpoint,
            # checkpoint_path="",
        )

    if 'cifar' in args.dataset:
        teacher = ptcv_get_model("resnet56_cifar10",pretrained=True)
        resume_checkpoint(
            teacher, args.resume,
            optimizer=None,
            loss_scaler=None,
            log_info=args.local_rank == 0)

    if args.quant_teacher:
        # args.wq_bitw = 4
        # args.wq_enable = True
        # args.aq_enable = False
        args.wq_bitw = 2
        # args.use_relu = False
        # args.aq_sym = True
        teacher = get_qat_model(teacher, args)
    if args.teacher_checkpoint != "":
        load_checkpoint(teacher, args.teacher_checkpoint, strict=True)
    teacher = teacher.eval()

    return teacher


def update_lr_multi_gpu(args):
    args.lr = args.world_size * args.lr
    args.min_lr = args.world_size * args.min_lr


def main(args):
    setup_default_logging(log_path='')

    args, args_text = args
    _logger.info(args_text)

    args.use_kd = args.use_kd or args.use_token_kd

    if args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else:
            _logger.warning("You've requested to log metrics to wandb but package not found. "
                            "Metrics not being logged to wandb, try `pip install wandb`")

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        # _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
        #              % (args.rank, args.world_size))
        update_lr_multi_gpu(args)
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d., lr %f' % (args.rank, args.world_size, args.lr))
    else:
        _logger.info('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    random_seed(args.seed, args.rank)

    if 'imagenet' in args.dataset:
        model = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            drop_rate=args.drop,
            drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
            drop_path_rate=args.drop_path,
            drop_block_rate=args.drop_block,
            global_pool=args.gp,
            # bn_tf=args.bn_tf,
            bn_momentum=args.bn_momentum,
            bn_eps=args.bn_eps,
            # scriptablse=args.torchscript,
            # checkpoint_path=args.initial_checkpoint,
            # use_distill_head=args.use_distill_head,
            # use_layer_scale=args.use_layer_scale,
            # use_skip=args.use_skip,
            # use_relu=args.use_relu,
            # use_bn=args.use_bn,
            # use_dual_skip=args.use_dual_skip,
            # down_block_type=args.down_block_type,
            # post_res_bn=args.post_res_bn,
            # use_softmax=True
        )

    if 'cifar' in args.dataset:
        #model = resnet_3stages.__dict__[args.model](num_classes=args.num_classes)
        model = MiniONN()

    print('Model modules:')
    for name, param in model.named_modules():
        print(name)
    
    print('Get QAT model...')
    model = get_qat_model(model, args)

    if args.initial_checkpoint != "":
        print('load checkpoint!')
        load_checkpoint(model, args.initial_checkpoint, strict=False)

    # for name, param in model.named_parameters(): 
    #     if ('conv' in name) and ('quan_a_fn.s' in name):
    #         print(name, '\t',param)
    # pdb.set_trace()

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly
    if args.replace_ln_by_bn:
        model = replace_ln_by_bn1d(model)
    _logger.info(f"Model {model}")

    # distillation
    teacher = None
    if args.use_kd:
        teacher = create_teacher_model(args)
        teacher.cuda()

    if args.local_rank == 0:
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        assert not args.split_bn
        if has_apex and use_amp != 'native':
            # Apex SyncBN preferred unless native amp is activated
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)

    # when creating optimizer, only weights and emb have weight-decay
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    resume_checkpoint(
            model, args.resume,
            optimizer=None,
            loss_scaler=None,
            log_info=args.local_rank == 0)

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp != 'native':
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.local_rank])  # can use device str in Torch >= 1.1
        # NOTE: EMA model does not need to be wrapped by DDP

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))   

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    
    if 'imagenet' in args.dataset:
        loader_train = create_loader(
            dataset_train,
            input_size=data_config['input_size'],
            batch_size=args.batch_size,
            is_training=True,
            use_prefetcher=args.prefetcher,
            no_aug=args.no_aug,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            re_split=args.resplit,
            scale=args.scale,
            ratio=args.ratio,
            hflip=args.hflip,
            vflip=args.vflip,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            num_aug_splits=num_aug_splits,
            interpolation=train_interpolation,
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=args.workers,
            distributed=args.distributed,
            collate_fn=collate_fn,
            pin_memory=args.pin_mem,
            use_multi_epochs_loader=args.use_multi_epochs_loader
        )
        _logger.info(f"Trainer transform: {loader_train.dataset.transform}")

        loader_eval = create_loader(
            dataset_eval,
            input_size=data_config['input_size'],
            batch_size=args.validation_batch_size_multiplier * args.batch_size,
            is_training=False,
            use_prefetcher=args.prefetcher,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=args.workers,
            distributed=args.distributed,
            crop_pct=data_config['crop_pct'],
            pin_memory=args.pin_mem,
        )
    
    # zwx: add this data processing
    if 'cifar10' in args.dataset:
        dataset_train = datasets.CIFAR10(
            root=args.data_dir,
            train=True,
            download = True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(size=32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=data_config['mean'], std=data_config['std']),
                transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),
            ])
        )
    
        dataset_eval = datasets.CIFAR10(
            root=args.data_dir,
            train=False,
            download = True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=data_config['mean'], std=data_config['std']),
            ]))
        
        loader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, \
            pin_memory=True, num_workers=args.workers, shuffle=True)      
        loader_eval = DataLoader(dataset=dataset_eval, batch_size=args.batch_size, \
            pin_memory=True, num_workers=args.workers, shuffle=False)

    # setup loss function
    if args.jsd:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing).cuda()
    elif args.use_token_kd:
        train_loss_fn = KLTokenMSELoss(alpha=args.kd_alpha, kd_type=args.kd_type)
    elif args.use_kd:
        train_loss_fn = KLLossSoft()
    elif mixup_active:
        # smoothing is handled with mixup target transform
        train_loss_fn = SoftTargetCrossEntropy().cuda()
    elif args.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()
    else:
        train_loss_fn = nn.CrossEntropyLoss().cuda()
    _logger.info(f"Train loss {train_loss_fn}")

    # if args.use_kd:
    #     train_loss_fn = KLLossSoft(train_loss_fn, alpha=args.kd_alpha)

    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    if args.use_kd:
        _logger.info("Verifying teacher model")
        validate(teacher, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)

    if args.initial_checkpoint != "":
        _logger.info("Verifying initial model")
        validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)

    # pdb.set_trace()
    
    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if args.rank == 0:
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                args.description,
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = get_outdir(args.output if args.output else './output/eval', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=args.checkpoint_hist)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)
        with open(os.path.join(output_dir, 'model.txt'), 'w') as f:
            f.write(str(model))

    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema,
                mixup_fn=mixup_fn, teacher=teacher)

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    _logger.info("Distributing BatchNorm running means and vars")
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                ema_eval_metrics = validate(
                    model_ema.module, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast,
                    log_suffix=' (EMA)')
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if output_dir is not None:
                update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None, log_wandb=args.log_wandb and has_wandb)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir=None, amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None, teacher=None):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            input, target = input.cuda(), target.cuda()
            output = model(input)

            if args.use_kd:
                target = teacher(input)
                loss = loss_fn(output, target)
            else:
                output = output[0] if isinstance(output, tuple) else output
                teacher = teacher[0] if isinstance(output, tuple) else output
                loss = loss_fn(output, target)

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            if args.local_rank == 0:
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])


def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                input, target = input.cuda(), target.cuda()
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))
            #pdb.set_trace()

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics


if __name__ == '__main__':
    args, args_text = parse_args()
    
    
    file = open("minionn_comm.txt")
    comm_dict = {}
    lines = file.readlines()
    for i in range(0,len(lines),2):
        if lines[i][0] == "#":
            continue
        percent = float(lines[i])
        setting = lines[i+1]
        setting = setting.split(' ')
        setting = [int(bit) for bit in setting]
        comm_dict[percent] = setting
    
    
    #for k,v in comm_dict.items():
    args.description = 'conv2'
    args.bit_setting = [2,8,8,8,8,8]
    main((args, args_text))
