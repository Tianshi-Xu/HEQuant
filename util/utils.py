import time
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, AverageMeter, dispatch_clip_grad, reduce_tensor
from timm.models import model_parameters
from contextlib import suppress
import torchvision
import os
from collections import OrderedDict
from src.quantization.modules.conv import QConvBn2d

def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, args, _logger,
        lr_scheduler=None, output_dir=None, loss_scaler=None, model_ema=None, mixup_fn=None, teacher=None, loss_fn_kd=None):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if mixup_fn is not None:
            mixup_fn.mixup_enabled = False
    reg_penalty = 0.

    def acc_reg_penalty(layer: QConvBn2d):
        """Accumulate the regularization penalty across constrained layers"""
        nonlocal reg_penalty
        scale_factor = layer.detach_bn_scaling_factor()
        scaled_weight_int = layer.weight * scale_factor
        # scaled_weight_int = scale_factor
        # print("torch.max(torch.abs(scaled_weight_int))", torch.max(torch.abs(scaled_weight_int)))
        total_bw_tensor = torch.log2(torch.sum(torch.abs(scaled_weight_int),dim=(1,2,3)))
        # # # If layer.quan_a_fn.bit is a Python scalar, it will be promoted in the sum.

        cur_penalty = total_bw_tensor.sum() # or scale_factor.detach()
        # cur_penalty = torch.sum(layer.weight ** 2)
        return cur_penalty

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    loader.sampler.set_epoch(epoch)
    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        input, target = input.cuda(), target.cuda()
        if mixup_fn is not None:
            input, target = mixup_fn(input, target)

        with torch.cuda.amp.autocast():
            input, target = input.cuda(), target.cuda()
            output = model(input)

            if args.use_kd:
                target = teacher(input)
                loss = loss_fn_kd(output, target)
            else:
                output = output[0] if isinstance(output, tuple) else output
                loss = loss_fn(output, target)
        # if args.local_rank == 0:
        #     _logger.info(f"loss: {loss.item()}, reg_penalty: {reg_penalty.item()*args.reg_weight}")
        # for layer in model.modules():
        #     if isinstance(layer, QConvBn2d):
        #         reg_penalty = reg_penalty + acc_reg_penalty(layer)
        # if args.local_rank == 0:
        #     _logger.info(f"loss: {loss.item()}, reg_penalty: {reg_penalty*args.reg_weight}")
        # loss = loss + args.reg_weight * reg_penalty
        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
        # if False:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad,
                parameters=model.parameters(),
                create_graph=False)
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
            optimizer.step()
        reg_penalty = 0
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

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)
            # lr_scheduler.step()

        end = time.time()
        # end for
    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()
    return OrderedDict([('loss', losses_m.avg)])


def validate(model, loader, loss_fn, args, _logger, log_suffix=''):
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
            input = input.cuda()
            target = target.cuda()

            with torch.amp.autocast(device_type='cuda'):
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

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics

