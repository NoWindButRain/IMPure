# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import webdataset as wds
from torch.utils.data import DataLoader
from torch.distributed.elastic.multiprocessing.errors import record

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

from recon_model.MAE.util import misc as misc
from recon_model.MAE.util.misc import NativeScalerWithGradNormCount as NativeScaler

import recon_model
import target_model

from dataset import transforms as ap_transforms
from dataset import AdvPurWDS, ConcatIterableDataset

from engine_finetune import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for adv pur', add_help=False)
    parser.add_argument('--debug', action='store_true',
                        help='debug')
    parser.set_defaults(debug=False)

    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--mul_epochs', default=3, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='iumae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--save_inter', default=2, type=int,
                        help='save interval')
    parser.add_argument('--feature_mask', default='mask', type=str,
                        help='feature mask, mask or noise')
    parser.add_argument('--img_mask', default='mask', type=str,
                        help='img mask, mask or noise')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--chunks', default=4, type=int,
                        help='chunks (percentage of removed patches).')
    parser.add_argument('--mask_chunks', default=4, type=int,
                        help='mask chunks (percentage of removed patches).')
    parser.add_argument('--recon_mean', action='store_true',
                        help='recon mean')
    parser.set_defaults(recon_mean=False)

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    parser.add_argument('--use_conv', action='store_true',
                        help='Use (per-patch) convk9c3s1')
    parser.set_defaults(use_conv=False)
    parser.add_argument('--norandom_mask', action='store_true',
                        help='no random mask')
    parser.set_defaults(norandom_mask=False)

    parser.add_argument('--autoweight_loss', action='store_true',
                        help='AutomaticWeightedLoss')
    parser.set_defaults(autoweight_loss=False)

    parser.add_argument('--hgds', default='[]', type=str,
                        help='Use hgds [{\"name\": \"resnet50_v2\", \"out_feature\": [-2]}]')
    parser.add_argument('--hgds_out', default='[]', type=str,
                        help='Use hgd out feature num')
    parser.add_argument('--val_modelname', default='resnet101_v2', type=str,
                        help='val model name')

    parser.add_argument('--freeze_dec', action='store_true',
                        help='freeze decoder')
    parser.set_defaults(freeze_dec=False)
    parser.add_argument('--freeze_enc', action='store_true',
                        help='freeze encoder')
    parser.set_defaults(freeze_enc=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-7, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-7, metavar='LR',
                        help='warmup learning rate (default: 1e-7)')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay_rate', type=float, default=0.5, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    parser.add_argument('--lr_cycle_mul', type=float, default=1., metavar='RATE',
                        help='cycle_mul (default: 1.0)')
    parser.add_argument('--lr_cycle_limit', type=int, default=3, metavar='N',
                        help='cycle_limit (default: 3)')
    parser.add_argument('--lr_cycle_decay', type=float, default=1., metavar='RATE',
                        help='cycle_decay (default: 1.0)')

    # Dataset parameters
    parser.add_argument('--adv_wds_data_urls', default=[], nargs='+', type=str,
                        help='adv webdataset data urls')
    parser.add_argument('--wds_data_totals', default=[], nargs='+', type=int,
                        help='webdataset data totals')
    parser.add_argument('--wds_data_url_nums', default=[], nargs='+', type=int,
                        help='webdataset data urls num')
    parser.add_argument('--train_batch_sizes', default=[64], nargs='+', type=int,
                        help='train_batch_sizes')

    parser.add_argument('--adv_wds_val_data_urls', default=[], nargs='+', type=str,
                        help='adv webdataset val data urls')
    parser.add_argument('--wds_val_data_total', default=50000, type=int,
                        help='webdataset val data total')
    parser.add_argument(
        '--val_batch_size', default=32, type=int,
        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    parser.add_argument('--adv_train', action='store_true',
                        help='adv_train')
    parser.set_defaults(adv_train=False)

    parser.add_argument('--random_crop', action='store_true',
                        help='random_crop')
    parser.set_defaults(random_crop=False)
    parser.add_argument('--crop_size', default=[224], nargs='+', type=int,
                        help='images random crop size')

    parser.add_argument('--output_dir', default='../checkpoints',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='../log_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--dmodel_resume', default='',
                        help='dmodel resume from checkpoint')

    parser.add_argument('--clean_feature', action='store_true')
    parser.set_defaults(clean_feature=False)


    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    transform_train = ap_transforms.Compose([
        # ap_transforms.Resize(
        #     [args.input_size, args.input_size], interpolation=3, antialias=True),
        ap_transforms.ToTensor(),
    ])
    transform_val = ap_transforms.Compose([
        # ap_transforms.Resize(
        #     [args.input_size, args.input_size], interpolation=3, antialias=True),
        ap_transforms.ToTensor(),
    ])
    _batch_sizes = [1]
    class rc_batchsampler:
        def __init__(self, crop_size, train_batch_size):
            self._trans_rand_rc = [
                [ap_transforms.RandomCrop([i, i], pad_if_needed=True), j]
                for i, j in zip(crop_size, train_batch_size)]

        def __call__(self, src):
            _rc, batch_size = self._trans_rand_rc[
                np.random.randint(len(self._trans_rand_rc))]
            batch = [0] * batch_size
            idx_in_batch = 0
            for sample in src:
                image, clean_image, label = sample[: 3]
                batch[idx_in_batch] = [*(_rc(image, clean_image)), torch.Tensor([label])]
                idx_in_batch += 1
                if idx_in_batch == batch_size:
                    batch = [torch.cat(
                        [batch[i][j].unsqueeze(0) for i in range(len(batch))], 0)
                        for j in range(len(batch[0]))]
                    yield batch
                    idx_in_batch = 0
                    _rc, batch_size = self._trans_rand_rc[
                        np.random.randint(len(self._trans_rand_rc))]
                    batch = [0] * batch_size
            # if idx_in_batch > 0:
            #     batch = [torch.cat(
            #         [batch[i][j].unsqueeze(0) for i in range(len(batch))], 0)
            #         for j in range(len(batch[0]))]
            #     yield batch[:idx_in_batch]

    url_start = 0
    _zip = zip(
        range(len(args.wds_data_url_nums)),
        args.wds_data_url_nums, args.wds_data_totals,
        args.train_batch_sizes,
        args.crop_size
    )
    dataset_train = []
    for wds_i, url_len, wds_data_total, train_batch_size, crop_size in _zip:
        adv_wds_data_urls = args.adv_wds_data_urls[url_start: url_start + url_len]
        url_start += url_len

        end_epoch = int(
            wds_data_total / misc.get_world_size() / args.num_workers /
            train_batch_size
        )
        length = end_epoch * args.num_workers
        _rbs = rc_batchsampler([crop_size], [train_batch_size]) if args.random_crop else None

        _dataset_train = AdvPurWDS(
            adv_wds_data_urls, transform=transform_train,
            seed=args.seed * 1000 + wds_i * 100000,
            length=length, end_epoch=end_epoch, epoch=args.start_epoch,
            rc_batchsampler= _rbs
        )
        dataset_train.append(_dataset_train)
    dataset_train = ConcatIterableDataset(dataset_train)

    val_end_epoch = int(
        args.wds_val_data_total / misc.get_world_size() / args.num_workers
    )
    val_length = val_end_epoch * args.num_workers
    dataset_val = AdvPurWDS(
        args.adv_wds_val_data_urls, transform=transform_val,
        length=val_length, end_epoch=0, debug=args.debug,
    )
    print('adv_wds train num: ', len(args.adv_wds_data_urls))
    print('adv_wds val num: ', len(args.adv_wds_val_data_urls))

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=None,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.val_batch_size*2,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    hgds = []
    hgds_hyper = []
    for hgd in json.loads(args.hgds):
        _hgd = getattr(target_model, hgd['name'])(
            normalize=True, return_nodes=hgd['out_feature'])
        hgds.append(_hgd.model)
        hgds_hyper.append(hgd['out_hyper'])

    val_model = getattr(target_model, args.val_modelname)(normalize=True)
    val_model.model = val_model.model.to(device)
    
    # define the model
    # TODO
    model = getattr(recon_model, args.model)(
        img_size=args.input_size,
        norm_pix_loss=args.norm_pix_loss,
        pre_norm=True,
        use_conv=args.use_conv,
        random_mask=not args.norandom_mask,
        hgds=hgds, hgds_hyper=hgds_hyper, chunks=args.chunks,
        mask_chunks=args.mask_chunks,
        feature_mask=args.feature_mask,
        img_mask=args.img_mask,
        dmodel_resume=args.dmodel_resume,
        clean_feature=args.clean_feature,
        recon_mean=args.recon_mean,
        autoweight_loss=args.autoweight_loss,
    ).to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = np.mean(args.train_batch_sizes) * \
        args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # freeze
    if args.freeze_enc:
        nofreeze_layers = [
            'decoder_head_embed', 'decoder_embed', 'mask_token',
            'decoder_blocks', 'decoder_norm', 'decoder_pred', 'decoder_rb1',
            'skip_mask_token', 'mid_blocks', 'decoder_skip_linear',
            'decoder_skip_head_linear'
        ]
        # freeze_layers = ['decoder_rb1']
        for name, param in model_without_ddp.named_parameters():
            param.requires_grad = False
            if any([name.startswith(_key) for _key in nofreeze_layers]):
                param.requires_grad = True

    if args.freeze_dec:
        nofreeze_layers = [
            'patch_embed', 'cls_token',
            'blocks', 'norm'
        ]
        for name, param in model_without_ddp.named_parameters():
            param.requires_grad = False
            if any([name.startswith(_key) for _key in nofreeze_layers]):
                param.requires_grad = True

    if len(hgds) != 0:
        for name, param in model_without_ddp.named_parameters():
            if name.startswith('hgds'):
                param.requires_grad = False

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(
        model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(
        param_groups, lr=args.lr, betas=(0.9, 0.95))
    # import torch_optimizer as optim
    # optimizer = optim.Lamb(param_groups, lr=args.lr, betas=(0.9, 0.95))
    # optimizer = torch.optim.SGD(param_groups, lr=args.lr)
    print(optimizer)

    from timm.scheduler import create_scheduler
    lr_scheduler, _ = create_scheduler(args, optimizer)
    print(lr_scheduler)

    loss_scaler = NativeScaler()

    misc.load_model(
        args=args, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs * args.mul_epochs):
        if args.debug:
            evaluate(
                model, data_loader_val, val_model.model, device, epoch, args)
        if args.distributed:
            for wds_i in range(len(data_loader_train.dataset.datasets)):
                _seed = epoch + args.seed * 1000 + wds_i * 100000
                _dataset = data_loader_train.dataset.datasets[wds_i]
                _dataset.pipeline[0].seed = _seed
                _dataset.pipeline[3].epoch = epoch

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, lr_scheduler, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % args.save_inter == 0 or epoch + 1 == args.epochs*args.mul_epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(
            model, data_loader_val, val_model.model, device, epoch, args)
        max_accuracy = max(max_accuracy, test_stats["recon_acc"])
        print(f'Max accuracy: {max_accuracy*100:.2f}%')

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            log_path = os.path.join(args.output_dir, "log.txt")
            with open(log_path, mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

@record
def _main():
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

if __name__ == '__main__':
    _main()
