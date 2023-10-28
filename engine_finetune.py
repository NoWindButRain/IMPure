# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch
from torch.nn import functional as F
from torchvision import transforms

from recon_model.MAE.util import misc as misc, lr_sched as lr_sched


def PGD(recon_model, target_model, x, y, eps=16/255, steps=5, ieps=4/255):
    recon_model.eval()
    target_model.eval()
    x_adv = x.detach()
    x_adv = x_adv + torch.zeros_like(x_adv).uniform_(-eps, eps)
    for _ in range(steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            x_pred = recon_model(x_adv)[1]
            x_pred = recon_model.module.unnormalize(x_pred)
            logits = target_model(x_pred)['classifier.6']
            loss = F.cross_entropy(logits, y.view(-1).long())
        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + ieps * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
        x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv


def train_one_epoch(
        model: torch.nn.Module,
        data_loader: Iterable, optimizer: torch.optim.Optimizer,
        lr_scheduler,
        device: torch.device, epoch: int, loss_scaler,
        log_writer=None,
        args=None):

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, data in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_scheduler.step(data_iter_step / len(data_loader) + epoch)
            # lr_sched.adjust_learning_rate(
            #     optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples, clean_samples, labels = data[: 3]
        samples = samples.to(device, non_blocking=True)
        clean_samples = clean_samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            if args.adv_train:
                adv_samples = PGD(model, model.module.hgds[0], samples, labels)
                optimizer.zero_grad()
                model.train(True)
                samples = torch.cat([samples, adv_samples], dim=0)
                clean_samples = torch.cat([clean_samples, clean_samples], dim=0)
                labels = torch.cat([labels, labels], dim=0)

            model.train(True)
            loss, _, show_loss = model(
                samples, clean_samples,
                labels=labels
            )

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        # metric_logger.update(loss=loss_value)
        metric_logger.update(**show_loss)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader: Iterable,
    val_model: torch.nn.Module,
    device: torch.device, epoch: int,
    args=None
):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test: [{}]'.format(epoch)

    inception_trans = transforms.Pad([0, 0, 21, 21])

    model.eval()
    val_model.eval()

    for batch in metric_logger.log_every(data_loader, 20, header):
        if args.debug:
            sample_url, sample_key = batch[3: 5]
            with open('../debug_file.txt', 'a') as file:
                for _url, _key in zip(sample_url, sample_key):
                    file.write(f'{_url}, {_key}\n')
        samples, clean_samples, labels = batch[: 3]
        if samples.shape[-1] == 299:
            samples = inception_trans(samples)
        samples = samples.to(device, non_blocking=True)
        clean_samples = clean_samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            loss, preds, show_loss = model(
                samples, labels=labels
            )
            if preds.shape[-1] == 320:
                preds = preds[:, :, : 299, : 299]
            show_loss['loss'] = model.module.pix_loss(
                preds,
                model.module.pre_normlize(clean_samples)
            ).mean()
            preds = model.module.unnormalize(preds)
            recon_out = val_model(preds)
            recon_acc = (recon_out.argmax(1) == labels).type(
                torch.float).sum().item() / len(labels)
            show_loss['recon_acc'] = recon_acc
            clean_out = val_model(clean_samples)
            clean_acc = (clean_out.argmax(1) == labels).type(
                torch.float).sum().item() / len(labels)
            show_loss['clean_acc'] = clean_acc
        metric_logger.update(**show_loss)

        # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}