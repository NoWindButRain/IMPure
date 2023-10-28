import os
import sys
import time
import json
import argparse

from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

import recon_model
import target_model
import advattack
from dataset import ImageNet

def get_args_parser():
    parser = argparse.ArgumentParser('adv attack', add_help=False)

    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    parser.add_argument('--model', default='resnet50_v1',
                        type=str, metavar='MODEL',
                        help='Name of model to attack')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--recon_model', default='rcmmff_idmae_vit_base_patch16_dec512d8b',
                        type=str, metavar='MODEL',
                        help='Name of recon model to attack')

    parser.add_argument('--attack', default='PGD',
                        type=str, metavar='MODEL',
                        help='Name of adv attack')
    parser.add_argument('--attack_param', default='{}',
                        type=str, metavar='MODEL',
                        help='Dict param of adv attack')
    parser.add_argument('--attack_path', default='/data/object_class/ILSVRC2012_attack',
                        type=str, metavar='MODEL',
                        help='adv attack image save path')

    parser.add_argument('--data_path', default='/data/object_class/ILSVRC2012',
                        type=str, metavar='MODEL',
                        help='imagenet path')
    parser.add_argument('--data_split', default='train',
                        type=str, metavar='MODEL',
                        help='train or val')
    parser.add_argument('--filter_data', default='',
                        type=str, metavar='MODEL',
                        help='filter data path')

    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')

    return parser


class RModel(torch.nn.Module):
    def __init__(self, rmodel, model):
        super(RModel, self).__init__()
        self.rmodel = rmodel
        self.model = model
        self.name = model.name
        
    def forward(self, x):
        x = self.rmodel(x)[1]
        x = self.rmodel.unnormalize(x)
        x = self.model(x)
        return x

def prepare_model(model, chkpt_dir, device):
    # load model
    checkpoint = torch.load(chkpt_dir, map_location=device)
    del_layers = [
        'pos_embed',
        'decoder_pos_embed',
        'dmodel.',
        'hgds.'
    ]
    _keys = []
    for _key in checkpoint['model'].keys():
        if any([_key.startswith(_lay) for _lay in del_layers]):
            _keys.append(_key)
    for _key in _keys:
        del checkpoint['model'][_key]
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    # print(msg)
    return model


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    model_name = args.model
    tmodel = getattr(target_model, model_name)(normalize=True)
    tmodel.model = tmodel.model.to(device)
    tmodel.model.name = tmodel.name
    tmodel.model.eval()
    
    if args.recon_model:
        chkpt_dir='/root/workspace/AdverPurifier/checkpoints/\
rcmmff_idmae_vit_base_patch16_dec512d8b_\
vgg19bn_v1_eval_38_51_norm1_random_crop_img_mask_fea_mask_at_3model_2/checkpoint-10.pth'
        rmodel = getattr(recon_model, 'rcmmff_idmae_vit_base_patch16_dec512d8b')(
            use_conv=True, random_mask=True,
            pre_norm=True, chunks=4, feature_mask='noise',
            img_mask='noise',
        ).to(device)
        rmodel = prepare_model(rmodel, chkpt_dir, device)
        rmodel.eval()

        model = RModel(rmodel, tmodel.model)
    else:
        model = tmodel.model
    
    criterion = nn.CrossEntropyLoss()

    batch_size = args.batch_size
    dataset_split = args.data_split  # 'train' 'val'
    # ImageNet_train_resnet50_v1_20230224234822.json ImageNet_val_resnet50_v1.json
    filter_data_path = args.filter_data
    root_path = args.data_path

    with open(filter_data_path, 'r') as file:
        filter_data = json.load(file)
    filter_data_dict = dict(filter_data)

    def valid_file(path):
        return path.replace(root_path, '.') in filter_data_dict


    dataset = ImageNet(
        root_path, transform=tmodel.preprocess, split=dataset_split,
        is_valid_file=valid_file
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    adv_path = args.attack_path
    # FGSM BIM PGD DIFGSM
    attack_name = args.attack
    attack_param = json.loads(args.attack_param)
    attack = getattr(advattack, attack_name)(
        model, dataloader, adv_path, criterion,
        **attack_param
    )
    print(model)
    print(dataset)
    print(attack.name)
    # attack.mask_ratio = 0.75
    # attack.patch_size = 8
    # attack.is_mask = True
    attack.forward()







