import os
import json
import sys
import argparse

from tqdm import tqdm

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import target_model
from dataset import ImageNet
from dataset import transforms as ap_transforms

from skimage.restoration import denoise_wavelet


def denoiser(img):
    img = img.numpy()
    img = denoise_wavelet(
        img, mode='soft', channel_axis=0,
        convert2ycbcr=True, method='BayesShrink'
    )
    img = torch.from_numpy(img)
    return img

def pixel_deflection_without_map(imgs, deflections=12000, window=20, wavelet_denoise=False):
    B, C, H, W = imgs.shape
    own_x = torch.randint(0, W, [deflections])
    own_y = torch.randint(0, H, [deflections])
    rand_x = torch.randint(-window, window, [deflections]) + own_x
    rand_y = torch.randint(-window, window, [deflections]) + own_y
    rand_x[rand_x < 0] = 0
    rand_x[rand_x > W-1] = W-1
    rand_y[rand_y < 0] = 0
    rand_y[rand_y > H - 1] = H - 1
    imgs[:, :, own_y, own_x] = imgs[:, :, rand_y, rand_x]

    if wavelet_denoise:
        imgs = torch.cat([denoiser(img).unsqueeze(0) for img in imgs], dim=0)
    return imgs

def get_args_parser():
    parser = argparse.ArgumentParser('model eval', add_help=False)

    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    parser.add_argument('--model', default='resnet101_v2',
                        type=str, metavar='MODEL',
                        help='Name of model to attack')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--adv_attack', action='store_true',
                        help='is adv attack')
    parser.set_defaults(adv_attack=False)
    parser.add_argument('--root_path', default='./',
                        type=str, metavar='MODEL',
                        help='data path')
    parser.add_argument('--meta_path', default='/data/object_class/ILSVRC2012',
                        type=str, metavar='MODEL',
                        help='meta path')

    parser.add_argument('--data_split', default='train',
                        type=str, metavar='MODEL',
                        help='train or val')
    parser.add_argument('--filter_data', default='',
                        type=str, metavar='MODEL',
                        help='filter data path')

    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')

    parser.add_argument('--random_resize_pad', action='store_true',
                        help='random_resize_pad')
    parser.set_defaults(random_resize_pad=False)

    parser.add_argument('--random_deflection', action='store_true',
                        help='random_deflection')
    parser.set_defaults(random_deflection=False)

    parser.add_argument('--wavelet_denoise', action='store_true',
                        help='wavelet_denoise')
    parser.set_defaults(wavelet_denoise=False)


    return parser


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"


    batch_size = args.batch_size
    is_adv_attack = args.adv_attack
    filter_data_path = args.filter_data
    if filter_data_path:
        with open(filter_data_path, 'r') as file:
            filter_data = json.load(file)
        filter_data_dict = dict(filter_data)
        if is_adv_attack:
            def valid_file(path):
                return path.replace(root_path, '.')[: -4] in filter_data_dict
        else:
            def valid_file(path):
                return path.replace(root_path, '.') in filter_data_dict
    else:
        def valid_file(path):
            return True


    # resnet50_v1 inceptionv3_v1 efficientnetv2l_v1
    model_name = args.model
    model = getattr(target_model, model_name)(normalize=is_adv_attack)
    model.model = model.model.to(device)

    root_path = args.root_path
    meta_path = args.meta_path
    dataset_split = args.data_split
    if is_adv_attack:
        _transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        _transform = model.preprocess

    if args.random_resize_pad:
        _transform = transforms.Compose([
            _transform,
            ap_transforms.RandomResizePad(310, 331, 331, 2, False),
        ])

    dataset = ImageNet(
        root_path, split=dataset_split, meta_path=meta_path,
        is_valid_file=valid_file, transform=_transform
    )

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    print(model)
    print(dataset)

    total_size = len(dataset)
    correct = 0
    with torch.no_grad():
        model.model.eval()
        for images, labels in tqdm(dataloader):
            if args.random_deflection:
                images = pixel_deflection_without_map(
                    images, wavelet_denoise=args.wavelet_denoise)
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model.model(images)
            preds = outputs.argmax(1)
            correct += (preds == labels).type(torch.float).sum().item()
    print(f"Test Error: Accuracy: {(100*correct/total_size):>0.2f}%")




