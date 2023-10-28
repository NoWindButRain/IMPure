import os
import json
import argparse

from tqdm import tqdm

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import recon_model
import target_model
from dataset import ImageNet
from advattack.attack import transform_invert


def get_args_parser():
    parser = argparse.ArgumentParser('purify adversarial samples ', add_help=False)

    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    parser.add_argument('--model', default='iumae_vit_large_patch16',
                        type=str, metavar='MODEL',
                        help='Name of model to purify')
    parser.add_argument('--feature_mask', default='mask', type=str,
                        help='feature mask, mask or noise')
    parser.add_argument('--img_mask', default='mask', type=str,
                        help='img mask, mask or noise')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--resume', default='./',
                        type=str, metavar='MODEL',
                        help='path of model checkpoint')
    parser.add_argument('--random_mask', action='store_true',
                        help='is random mask')
    parser.set_defaults(random_mask=False)
    parser.add_argument('--chunks', default=4, type=int,
                        help='chunks (percentage of removed patches).')

    parser.add_argument('--adv_attack', action='store_true',
                        help='is adv attack')
    parser.set_defaults(adv_attack=False)
    parser.add_argument('--val_preprocess', action='store_true',
                        help='input val model need pad and resize')
    parser.set_defaults(val_preprocess=False)

    parser.add_argument('--root_path', default='./',
                        type=str, metavar='MODEL',
                        help='data path')
    parser.add_argument('--meta_path', default='/data/object_class/ILSVRC2012',
                        type=str, metavar='MODEL',
                        help='meta path')
    parser.add_argument('--save_path', default='../',
                        type=str, metavar='MODEL',
                        help='save path')

    parser.add_argument('--target_model', default='resnet101_v2', type=str,
                        help='resnet101_v2 inceptionv3_v1 inceptionresnetv2_v1')
    parser.add_argument('--val_model', action='store_true',
                        help='val model')
    parser.set_defaults(val_model=False)
    parser.add_argument('--data_split', default='val',
                        type=str, metavar='MODEL',
                        help='train or val')
    parser.add_argument('--filter_data', default='',
                        type=str, metavar='MODEL',
                        help='filter data path')

    parser.add_argument('--save_img', action='store_true',
                        help='save puri img')
    parser.set_defaults(save_img=False)

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
    batch_size = args.batch_size

    root_path = args.root_path
    # root_path = '/data/object_class/ILSVRC2012'
    meta_path = args.meta_path
    dataset_split = args.data_split # 'train' 'val'
    filter_data_path = args.filter_data
    with open(filter_data_path, 'r') as file:
        filter_data = json.load(file)
    filter_data_dict = dict(filter_data)
    is_adv_attack = args.adv_attack
    if is_adv_attack:
        def valid_file(path):
            return path.replace(root_path, '.')[: -4] in filter_data_dict

        preprocesses = {
            'resnet101_v2': transforms.Compose([
                transforms.ToTensor()
            ]),
            'inceptionv3_v1': transforms.Compose([
                transforms.Pad([0, 0, 21, 21]),
                transforms.ToTensor()
            ]),
            'inceptionresnetv2_v1': transforms.Compose([
                transforms.Pad([0, 0, 21, 21]),
                transforms.ToTensor()
            ])
        }
        preprocess = preprocesses[args.target_model]
    else:
        def valid_file(path):
            return path.replace(root_path, '.') in filter_data_dict

        preprocesses = {
            'resnet101_v2': transforms.Compose([
                transforms.Resize([232], interpolation=2, antialias=None),
                transforms.CenterCrop([224]),
                transforms.ToTensor()
            ]),
            'inceptionv3_v1': transforms.Compose([
                transforms.Resize([342], interpolation=2, antialias=None),
                transforms.CenterCrop([299]),
                transforms.Pad([0, 0, 21, 21]),
                transforms.ToTensor()
            ]),
            'inceptionresnetv2_v1': transforms.Compose([
                transforms.Resize([333], interpolation=3, antialias=None),
                transforms.CenterCrop([299]),
                transforms.Pad([0, 0, 21, 21]),
                transforms.ToTensor()
            ]),
        }
        preprocess = preprocesses[args.target_model]
    dataset = ImageNet(
        root_path, dataset_split, is_valid_file=valid_file, meta_path=meta_path,
        transform=preprocess
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False
    )

    chkpt_dir = args.resume
    model_name = args.model
    use_conv = True
    # build model
    model = getattr(recon_model, model_name)(
        use_conv=use_conv, random_mask=args.random_mask,
        pre_norm=True, chunks=args.chunks, feature_mask=args.feature_mask,
        img_mask=args.img_mask,
    ).to(device)
    model = prepare_model(model, chkpt_dir, device)
    model.eval()
    print('Model loaded.')
    save_path = args.save_path

    if args.val_model:
        t_model = getattr(target_model, args.target_model)(normalize=True)
        t_model.model = t_model.model.to(device)
        t_model.model.eval()

        val_preprocesses = {
            'resnet101_v2': transforms.Compose([
                transforms.Pad(32),
                transforms.Resize((224, 224), antialias=None),
            ]),
            'inceptionv3_v1': transforms.Compose([
                transforms.Resize((331, 331), antialias=None),
                # transforms.Pad(8),
            ]),
            'inceptionresnetv2_v1': transforms.Compose([
                # transforms.Pad(16),
                transforms.Resize((331, 331), antialias=None),
            ]),
        }
        val_preprocess = val_preprocesses[args.target_model]

    total_size = len(dataset)
    correct = 0
    with torch.no_grad():
        model.eval()
        t_model.model.eval()
        for i, (images, labels) in enumerate(tqdm(dataloader)):
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(images)[1]
            if not use_conv:
                outputs = model.unpatchify(outputs)
            if outputs.shape[-1] == 320:
                outputs = outputs[:, :, : 299, : 299]
            outputs = model.unnormalize(outputs)

            if args.val_model:
                if args.val_preprocess:
                    outputs = val_preprocess(outputs)
                with torch.cuda.amp.autocast():
                    recon_out = t_model.model(outputs)
                correct += (recon_out.argmax(1) == labels).type(
                    torch.float).sum().item()

            outputs = outputs.detach().float().cpu()
            if args.save_img:
                for j, output in enumerate(outputs):
                    _image = transform_invert(output)
                    _index = i * dataloader.batch_size + j
                    _save_path = dataset.imgs[_index][0].replace(
                        dataset.root, save_path)
                    os.makedirs(os.path.split(_save_path)[0], exist_ok=True)
                    _image.save(_save_path, quality=100, subsampling=0)
        print('Acc: ', correct / total_size)

if __name__ == '__main__':
    main()
