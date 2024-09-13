# Additive Power-of-Two Quantization: An Efficient Non-uniform Discretization For Neural Networks
# Yuhang Li, Xin Dong, Wei Wang
# International Conference on Learning Representations (ICLR), 2020.


import argparse
import os
import random
import shutil
import time
import warnings
import yaml
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from logging.handlers import RotatingFileHandler

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from src.quantization.modules.conv import QConv2d, QConvBn2d
from src import *
from src.quantization.utils import KLLossSoft
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, \
     model_parameters
from timm.models.layers import convert_splitbn_model
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler
from tensorboardX import SummaryWriter
import sys
import gc

_logger = logging.getLogger('train')
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--model', default='resnet101', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('-j','--workers', default=4, type=int, metavar='N',help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1024, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--data', metavar='DATA_PATH', default='./data/',
                    help='path to imagenet data (default: ./data/)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=30, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
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
parser.add_argument('--bw_list', default="", type=str,
                    help='loool')
parser.add_argument('--ba_list', default="", type=str,
                    help='loool')
best_acc1 = 0
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

def create_teacher_model(args):
    teacher = create_model(
        args.teacher,
        pretrained=False,
        num_classes=args.num_classes,
        drop_rate=0.,
        drop_connect_rate=0.,
        drop_block_rate=0.,
        # checkpoint_path=args.teacher_checkpoint,
        # checkpoint_path="",
    )
    if args.teacher_checkpoint != "":
        load_checkpoint(teacher, args.teacher_checkpoint, strict=False)
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
                "symmetric": True,  # not args.wq_asym,
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

def main():
    args, args_text = _parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    setup_default_logging()
    handler = RotatingFileHandler(args.log_name+'.log', maxBytes=10*1024*1024, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    _logger.addHandler(handler)
    
    args.gpu = gpu
    if args.gpu == 0:
        _logger.info(f'bath_size: {args.batch_size}')
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    teacher = None
    if args.use_kd:
        teacher = create_teacher_model(args)
    model = create_model(
        args.model,
        num_classes=args.num_classes)
    _logger.info(args.model)
    if args.initial_checkpoint:
        load_checkpoint(model, args.initial_checkpoint,strict=False)
    model = get_qat_model(model, args)
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            if args.use_kd:
                teacher.cuda(args.gpu)
            validate_loss_fn = nn.CrossEntropyLoss().cuda(args.gpu)
            train_loss_fn_kd = KLLossSoft().cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            # teacher = torch.nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            if args.use_kd:
                teacher.cuda()
                # teacher = torch.nn.parallel.DistributedDataParallel(teacher)
            validate_loss_fn = nn.CrossEntropyLoss().cuda()
            train_loss_fn_kd = KLLossSoft().cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # init from pre-trained model or full-precision model

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    # customize the lr and wd for clipping thresholds
    # model_params = []
    # for name, params in model.module.named_parameters():
    #     if 'act_alpha' in name:
    #         model_params += [{'params': [params], 'lr': 3e-2, 'weight_decay': 2e-5}]
    #     elif 'wgt_alpha' in name:
    #         model_params += [{'params': [params], 'lr': 1e-2, 'weight_decay': 1e-4}]
    #     else:
    #         model_params += [{'params': [params]}]
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1e+6))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # data loader by official torchversion:
    # --------------------------------------------------------------------------
    print('==> Using Pytorch Dataset')
    input_size = 224  # image resolution for resnets
    import torchvision
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    torchvision.set_image_backend('accimage')
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    # --------------------------------------------------------------------------
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    writer = SummaryWriter(comment='res18_4bit')
    if args.use_kd:
        _logger.info("Verifying teacher model")
        # validate(val_loader, teacher, validate_loss_fn, args)
    if args.gpu == 0:
        _logger.info(model)
    if args.initial_checkpoint != "":
        _logger.info("Verifying initial model in training dataset")
        # validate(val_loader, model, validate_loss_fn, args)
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
    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, writer,teacher=teacher, loss_fn_kd=train_loss_fn_kd)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)
        scheduler.step()
        writer.add_scalar('test_acc', acc1, epoch)
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        print('best_acc:' + str(best_acc1))

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args, writer, teacher, loss_fn_kd):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        output = model(images)
        if args.use_kd:
            teacher_target = teacher(images)
            loss = loss_fn_kd(output, teacher_target)
        else:
        # compute output
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        gc.collect()
    writer.add_scalar('train_acc', top1.avg, epoch)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    # model.module.show_params()

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoints.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'res18_4best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    adjust_list = [30,60,80,100]
    if epoch in adjust_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()