import time
from itertools import zip_longest

import torch
from torch.utils.data import IterableDataset
from torchvision import datasets
from torchvision import transforms
import webdataset as wds

from . import ImageNet
from . import base_wds
from . import transforms as ap_transforms


preprocess = ap_transforms.Compose([
    ap_transforms.CenterCrop([224]),
    ap_transforms.ToTensor(),
    ap_transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225)
    ),
])


class AdvPurDataset(ImageNet):
    def __init__(
            self, adv_root, clean_root,
            transform=preprocess, adv_img=True,
            **kwargs
    ):
        super(AdvPurDataset, self).__init__(adv_root, meta_path=clean_root, **kwargs)
        self.adv_root = adv_root
        self.clean_root = clean_root
        self.transform = transform
        self.adv_img = adv_img

    def __getitem__(self, index):
        path, target = self.samples[index]
        adv_sample = self.loader(path)
        clean_path = path.replace(self.adv_root, self.clean_root)
        if self.adv_img:
            clean_path = clean_path[: -4]
        clean_sample = self.loader(clean_path)
        if self.transform is not None:
            adv_sample, clean_sample = self.transform(adv_sample, clean_sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return adv_sample, clean_sample, target


def AdvPurWDS(
        urls, transform=preprocess, **kwargs
):
    map_key = ["adv.png", "clean.jpg", "cls"]
    def map_tran(sample):
        adv_sample, clean_sample, target = sample[: 3]
        adv_sample, clean_sample = transform(adv_sample, clean_sample)
        return adv_sample, clean_sample, int(target), *sample[3:]

    dataset = base_wds(
        urls, map_key, map_tran, **kwargs
    )
    dataset.name = 'AdvPurWDS'

    return dataset


class ConcatIterableDataset(IterableDataset):
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatIterableDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __iter__(self):
        for items in zip_longest(*self.datasets):
            for item in items:
                if item is not None:
                    yield item


if __name__ == '__main__':
    from dataset import transforms as ap_transforms

    transform_train = ap_transforms.Compose([
        ap_transforms.Resize([224, 224], interpolation=3),
        ap_transforms.RandomHorizontalFlip(),
        ap_transforms.ToTensor(),
    ])

    # _ap = '/data/object_class/adverpuri/attack/inceptionv3_v1_FGSM_4'
    # _cp = '/data/object_class/ILSVRC2012'
    # dataset = AdvPurDataset(_ap, _cp, split='val', transform=transform_train)
    # print(len(dataset))
    # adv_sample, clean_sample, target = dataset[0]
    # print(adv_sample.size(), clean_sample.size(), target)

    urls = [
        '/data/object_class/adverpuri/attack_wds/resnet101_v2_FGSM_8/train/000071.tar',
        '/data/object_class/adverpuri/attack_wds/resnet101_v2_FGSM_8/train/000071.tar'
    ]
    dataset = AdvPurWDS(urls, transform_train, end_epoch=400, debug=True)
    print(dataset.pipeline)
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=2, batch_size=32, pin_memory=True)
    for epoch in range(4):
        dataloader.dataset.pipeline[0].epoch = epoch
        dataloader.dataset.pipeline[2].epoch = epoch
        for i, (adv_sample, clean_sample, labels, _urls, _keys) in enumerate(dataloader):
            if i == 0:
                print(adv_sample.size(), labels, _urls, _keys)








