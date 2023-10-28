import os
import json
import argparse

from PIL import Image

import torch
from torchvision import transforms

import recon_model
import target_model
from advattack.attack import transform_invert


def get_args_parser():
    parser = argparse.ArgumentParser('IMPure Demo', add_help=False)

    parser.add_argument('--model', default='rcmmff_idmae_vit_base_patch16_dec512d8b',
                        type=str, metavar='MODEL',
                        help='Name of model to purify')
    parser.add_argument('--feature_mask', default='noise', type=str,
                        help='feature mask, mask or noise')
    parser.add_argument('--img_mask', default='noise', type=str,
                        help='img mask, mask or noise')
    parser.add_argument('--resume', default='./checkpoints/mipure.pth',
                        type=str, metavar='MODEL',
                        help='path of model checkpoint')
    parser.add_argument('--chunks', default=4, type=int,
                        help='chunks (percentage of removed patches).')

    parser.add_argument('--img_path', default='./demo/1.jpg',
                        type=str, metavar='MODEL',
                        help='path of input image')
    parser.add_argument('--save_path', default='./demo/1_recon.jpg',
                        type=str, metavar='MODEL',
                        help='save path')

    parser.add_argument('--target_model', default='resnet101_v2', type=str,
                        help='resnet101_v2 inceptionv3_v1 inceptionresnetv2_v1')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')

    return parser


def prepare_model(model, chkpt_dir, device):
    # load model
    checkpoint = torch.load(chkpt_dir, map_location=device)
    del_layers = [
        'pos_embed',
        'decoder_pos_embed',
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


def main():
    args = get_args_parser()
    args = args.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    preprocess = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size), antialias=None),
        transforms.ToTensor()
    ])

    model = getattr(recon_model, args.model)(
        use_conv=True, pre_norm=True, chunks=args.chunks,
        feature_mask=args.feature_mask, img_mask=args.img_mask,
    ).to(device)
    model = prepare_model(model, args.resume, device)
    model.eval()
    print('Recon Model loaded.')

    t_model = getattr(target_model, args.target_model)(normalize=True)
    t_model.model = t_model.model.to(device)
    t_model.model.eval()
    print('Target Model loaded.')

    image = Image.open(args.img_path)
    image = preprocess(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            out = t_model.model(image)
            class_out = out.argmax(1)[0].item()
            print('Classification result: ', class_out)
            outputs = model(image)[1]
            outputs = model.unnormalize(outputs)
            recon_out = t_model.model(outputs)
            class_out = recon_out.argmax(1)[0].item()
            print('Classification result of purified samples: ', class_out)

        output = outputs.detach().float().cpu()[0]
        _image = transform_invert(output)
        _image.save(args.save_path, quality=100, subsampling=0)


if __name__ == '__main__':
    main()




