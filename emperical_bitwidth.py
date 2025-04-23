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
import sys
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP
# sys.path.append("/data/home/menglifrl/pytorch-image-models")
sys.path.append("/home/xts/code/MySparsity/pytorch-image-models")

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
parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
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
parser.add_argument('--use-relu', action='store_true', default=False,
                    help='whether to use relu for the model')
parser.add_argument('--replace-relu', action='store_true', default=False,
                    help='whether to replace relu with prelu for the model')
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
parser.add_argument('--distributed', action='store_true', default=False,
                    help='whether to use distributed training')

# sparsity
parser.add_argument('--ws-enable', action='store_true', default=False,
                    help='enable weight sparsification')
parser.add_argument('--ws-mode', default='Identity', type=str,
                    help='weight sparsification mode')
parser.add_argument('--ws-patterns', default='1:1', type=str,
                    help='weight sparsification pattern')
parser.add_argument('--as-enable', action='store_true', default=False,
                    help='enable act sparsification')
parser.add_argument('--as-mode', default='Identity', type=str,
                    help='act sparsification mode')
parser.add_argument('--as-patterns', default='1:1', type=str,
                    help='act sparsification pattern')


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

    def _decode_args(m):
        if ";" not in m:
            return m, {}, {}
        args = m.split(";")
        name = args[0]
        ret = {"wq": {}, "aq": {}}
        for arg in args[1:]:
            val = arg.split(":")
            assert val[1] not in ret[val[0]]
            ret[val[0]][val[1]] = eval(val[2])
        return name, ret["wq"], ret["aq"]

    qat_model = copy.deepcopy(model)
    qat_model.train()
    qconfigs = {}
    for m in args.qmodules:
        mod, wq, aq = _decode_args(m)
        wqcfg = {
            "enable": args.wq_enable,
            "mode": args.wq_mode if args.wq_enable else "Identity",
            "bit": args.wq_bitw,
            "thd_pos": args.wq_pos,
            "thd_neg": args.wq_neg,
            "all_positive": False,
            "symmetric": not args.wq_asym,
            "per_channel": args.wq_per_channel,
            "normalize_first": False,
        }
        wqcfg.update(wq)
        aqcfg = {
            "enable": args.aq_enable,
            "mode": args.aq_mode if args.aq_enable else "Identity",
            "bit": args.aq_bitw,
            "thd_pos": args.aq_pos,
            "thd_neg": args.aq_neg,
            "all_positive": True if args.use_relu and "linear2" in m else False,
            "symmetric": not args.aq_asym,
            "per_channel": False,
            "normalize_first": False,
        }
        aqcfg.update(aq)
        wscfg = {
            "enable": args.ws_enable,
            "mode": args.ws_mode if args.ws_enable else "Identity",
            "patterns": args.ws_patterns,
        }
        ascfg = {
            "enable": args.as_enable,
            "mode": args.as_mode if args.as_enable else "Identity",
            "patterns": args.as_patterns,
        }
        qconfigs[mod] = {"weight_q": wqcfg, "act_q": aqcfg, "weight_s": wscfg, "act_s": ascfg}
    qat_model = replace_module_by_qmodule(model, qconfigs)

    qconfigs = {}
    for m in args.resq_modules:
        acfg = {
            "enable": args.resq_enable,
            "mode": args.resq_mode if args.resq_enable else "Identity",
            "bit": args.resq_bitw,
            "thd_pos": args.resq_pos,
            "thd_neg": args.resq_neg,
            "all_positive": args.use_relu,
            "symmetric": not args.resq_asym,
            "per_channel": False,
            "normalize_first": False,
        }
        qconfigs[m] = {"act_q": acfg}
    qat_model = register_act_quant_hook(model, qconfigs)
    return qat_model


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

    # import pdb; pdb.set_trace()

    args.use_kd = args.use_kd or args.use_token_kd

    if args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else:
            _logger.warning("You've requested to log metrics to wandb but package not found. "
                            "Metrics not being logged to wandb, try `pip install wandb`")

    args.prefetcher = not args.no_prefetcher
    # args.distributed = False
    # if 'WORLD_SIZE' in os.environ:
    #     args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
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

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        # checkpoint_path=args.initial_checkpoint,
        use_distill_head=args.use_distill_head,
        use_layer_scale=args.use_layer_scale,
        use_skip=args.use_skip,
        use_relu=args.use_relu,
        use_bn=args.use_bn,
        use_dual_skip=args.use_dual_skip,
        down_block_type=args.down_block_type,
        post_res_bn=args.post_res_bn,
        use_softmax=True)
    if args.replace_relu:
        model = replace_relu_by_prelu(model)
    model = get_qat_model(model, args)
    if args.initial_checkpoint != "":
        # print("OK")
        incompatible_keys = load_checkpoint(model, args.initial_checkpoint, strict=False)

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
    # cal(model)
    # exit(0)
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
    if args.resume:
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)
        # optimizer.param_groups[1]['weight_decay'] = 0
        # optimizer.param_groups[0]['weight_decay'] = 0
        _logger.info(f"{optimizer}")

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

    # create the train and eval datasets
    dataset_train = create_dataset(
        args.dataset,
        root=args.data_dir, split=args.train_split, is_training=True,
        batch_size=args.batch_size, repeats=args.epoch_repeats, download=True)
    dataset_eval = create_dataset(
        args.dataset, root=args.data_dir, split=args.val_split, is_training=False, batch_size=args.batch_size)

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

    # create data loaders w/ augmentation pipeiine
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
    _logger.info(f"Validate transform: {loader_eval.dataset.transform}")

    # if args.use_kd:
    #     train_loss_fn = KLLossSoft(train_loss_fn, alpha=args.kd_alpha)

    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    if args.use_kd:
        _logger.info("Verifying teacher model")
        validate(teacher, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)

    if args.initial_checkpoint != "":
        _logger.info("Verifying initial model")
        validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)

    # setup checkpoint saver and eval metric tracking
    best_metric = None
    best_epoch = None
            
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

def worker(idx, batch_size, x_q, w_q, padded_x_q, results):
    part_sum = torch.zeros_like(padded_x_q[0, :, :, :])
    max_max_num, max_min_num, max_mean_num = 0, 0, 0
    max_bit, min_bit, mean_bit = 0, 0, 0
    # print("w_q_unique:",w_q.unique())
    # padded_x_q_cpu = padded_x_q.cpu()
    # w_q_cpu = w_q.cpu()
    # print(x_q.unique())
    for n in range(idx * batch_size, min((idx + 1) * batch_size, x_q.shape[0])): 
        # print("n:",n)
        for k in range(w_q.shape[0]):
            for w in range(x_q.shape[2]):
                for h in range(x_q.shape[3]):
                    if w_q.shape[1] == 1:
                        part_sum = torch.mul(padded_x_q[n, k, w:w+w_q.shape[2], h:h+w_q.shape[3]], w_q[k, 0, :, :])
                    else:
                        # print("kernel size:",w_q.shape[2],w_q.shape[3])
                        # sys.stdout.flush()
                        part_sum = torch.mul(padded_x_q[n, :, w:w+w_q.shape[2], h:h+w_q.shape[3]], w_q[k, :, :, :])
                    # print("part_sum: ",part_sum)
                    tmp_max_mean_num = torch.abs(part_sum.flatten().cumsum(0)).max()
                    tmp_mean_bit = torch.log2(tmp_max_mean_num).ceil()
                    mean_bit = max(mean_bit, tmp_mean_bit)
                    max_mean_num = max(max_mean_num, tmp_max_mean_num)
                    # print("mean_bit: ",mean_bit)
                    negative_part_sum = part_sum[part_sum < 0]
                    positive_part_sum = part_sum[part_sum > 0]

                    tmp_max_max_num = torch.sum(torch.abs(negative_part_sum))
                    tmp_max_max_num = max(tmp_max_max_num, torch.sum(positive_part_sum))
                    tmp_max_bit = torch.log2(tmp_max_max_num).ceil()
                    max_bit = max(max_bit, tmp_max_bit)
                    max_max_num = max(max_max_num, tmp_max_max_num)

                    tmp_max_min_num = torch.max(torch.abs(part_sum))
                    tmp_max_min_num = max(tmp_max_min_num, part_sum.sum())
                    tmp_min_bit = torch.log2(tmp_max_min_num).ceil()
                    min_bit = max(min_bit, tmp_min_bit)
                    max_min_num = max(max_min_num, tmp_max_min_num)
                    
    print("done:", idx)
    
    results = max(results, [max_max_num, max_min_num, max_mean_num, max_bit, min_bit, mean_bit])
    print(results)
    sys.stdout.flush()

def check_range(model, inp, out):
    if model.stride[0]==2:
        return
    # if model.groups != 1:
    #     return
    try:
        mp.set_start_method('spawn', force=True)
        print("spawned")
    except RuntimeError:
        pass
    print("In layer: ",model)
    sys.stdout.flush()
    new_inp = inp[0].clone().detach()
    x_q = model.quan_a_fn(new_inp)
    x_q_alpha = model.quan_a_fn.alpha
    w_q = model.quan_w_fn(model.weight)
    w_q_alpha = model.quan_w_fn.alpha
    x_q = x_q / x_q_alpha
    w_q = w_q / w_q_alpha
    
    padded_x_q = F.pad(x_q, (1, 1, 1, 1))
    num_processes = mp.cpu_count()*3//4  # 获取可用的CPU核心数
    batch_size = (x_q.shape[0] + num_processes - 1) // num_processes  # 每个进程处理的样本数
    print("batch:",batch_size)
    # 创建进程池和共享内存数组
    pool = mp.Pool(processes=num_processes)
    results = [0, 0, 0, 0, 0, 0]
    
    # 调用worker函数进行并行计算
    jobs = []
    for i in range(num_processes):
        job = pool.apply_async(worker, (i, batch_size, x_q, w_q, padded_x_q, results))
        jobs.append(job)
    
    # 等待所有任务完成
    for job in jobs:
        job.get()
    
    # 汇总计算结果
    print("max num: ",results[0])
    print("min num: ",results[1])
    print("mean num: ",results[2])
    print("max_bit: ",results[3])
    print("min_bit: ",results[4])
    print("mean_bit: ",results[5])


def check_w_l1norm(model, inp, out):
    # if model.stride[0]==2:
    #     return
    print("In layer: ",model)
    sys.stdout.flush()
    new_inp = inp[0].clone().detach()
    x_q = torch.ones_like(new_inp)
    w_q = model.quan_w_fn(model.weight)
    w_q_alpha = model.quan_w_fn.alpha
    w_q = w_q / w_q_alpha
    # print(w_q)
    y_q = F.conv2d(x_q, torch.abs(w_q), stride=model.stride, padding=model.padding, dilation=model.dilation, groups=model.groups)
    print("max l1 norm: ",torch.max(torch.abs(y_q)))
    print("max l1 norm bit:",torch.log2(torch.max(torch.abs(y_q))).ceil())
    sys.stdout.flush()
    
def hook(model,fn):
    for name, module in model.named_modules():
        if isinstance(module, QConv2d):
            module.register_forward_hook(fn)

def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
    # hook(model, check_w_l1norm)
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    hook(model,check_range)
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
                    'Time: {batch_time.val:.3f} ({batch_time.sum:.3f}s) '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))
            print("total ok")
            exit(0)
    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics


if __name__ == '__main__':
    args, args_text = parse_args()
    main((args, args_text))