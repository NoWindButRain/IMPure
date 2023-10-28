import random

import torch
from torchvision import transforms
from torchvision.transforms import functional as F


class Resize(transforms.Resize):
    def forward(self, img, *mask):
        if len(mask) == 0 or mask[0] is None:
            return super(Resize, self).forward(img)
        _img = F.resize(img, self.size, self.interpolation, self.max_size, self.antialias)
        _mask = [F.resize(m, self.size, self.interpolation, self.max_size, self.antialias) for m in mask]
        return _img, *_mask


class CenterCrop(transforms.CenterCrop):
    def forward(self, img, *mask):
        if len(mask) == 0 or mask[0] is None:
            return super(CenterCrop, self).forward(img)
        _img = F.center_crop(img, self.size)
        _mask = [F.center_crop(m, self.size) for m in mask]
        return _img, *_mask


class Pad(transforms.Pad):
    def forward(self, img, *mask):
        if len(mask) == 0 or mask[0] is None:
            return super(Pad, self).forward(img)
        _img = F.pad(img, self.padding, self.fill, self.padding_mode)
        _mask = [F.pad(m, self.padding, self.fill, self.padding_mode) for m in mask]
        return _img, *_mask


class RandomApply(transforms.RandomApply):
    def forward(self, img, *mask):
        if len(mask) == 0 or mask[0] is None:
            return super(RandomApply, self).forward(img)
        if self.p < torch.rand(1):
            return img, *mask
        for t in self.transforms:
            img, *mask = t(img, *mask)
        return img, *mask

class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def forward(self, img, *mask):
        if len(mask) == 0 or mask[0] is None:
            return super(RandomHorizontalFlip, self).forward(img)
        if torch.rand(1) < self.p:
            return F.hflip(img), *[F.hflip(m) for m in mask]
        return img, *mask


class RandomCrop(transforms.RandomCrop):
    def forward(self, img, *mask):
        if len(mask) == 0 or mask[0] is None:
            return super(RandomCrop, self).forward(img)
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            mask = [F.pad(m, self.padding, self.fill, self.padding_mode) for m in mask]
        _, height, width = F.get_dimensions(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
            mask = [F.pad(m, padding, self.fill, self.padding_mode) for m in mask]
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)
            mask = [F.pad(m, padding, self.fill, self.padding_mode) for m in mask]

        i, j, h, w = self.get_params(img, self.size)
        img = F.crop(img, i, j, h, w)
        mask = [F.crop(m, i, j, h, w) for m in mask]
        return img, *mask


class RandomResizedCrop(transforms.RandomResizedCrop):
    def forward(self, img, *mask):
        if len(mask) == 0 or mask[0] is None:
            return super(RandomResizedCrop, self).forward(img)
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        _img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation, antialias=self.antialias)
        _mask = [F.resized_crop(m, i, j, h, w, self.size, self.interpolation, antialias=self.antialias) for m in mask]
        return _img, *_mask


class ToTensor(transforms.ToTensor):
    def __call__(self, pic, *mask):
        if len(mask) == 0 or mask[0] is None:
            return super(ToTensor, self).__call__(pic)
        return F.to_tensor(pic), *[F.to_tensor(m) for m in mask]


class Normalize(transforms.Normalize):
    def forward(self, tensor, *mask):
        if len(mask) == 0 or mask[0] is None:
            return super(Normalize, self).forward(tensor)
        _tensor = F.normalize(tensor, self.mean, self.std, self.inplace)
        _mask = [F.normalize(m, self.mean, self.std, self.inplace) for m in mask]
        return _tensor, *_mask


class Compose(transforms.Compose):
    def __call__(self, img, *mask):
        if len(mask) == 0 or mask[0] is None:
            return super(Compose, self).__call__(img)
        for t in self.transforms:
            img, *mask = t(img, *mask)
        return img, *mask


class RandomResizePad(torch.nn.Module):
    def __init__(self, l_size, h_size, end_size, interpolation=2, antialias='warn'):
        super(RandomResizePad, self).__init__()
        self.l_size = l_size
        self.h_size = h_size
        self.end_size = end_size
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img):
        size = random.randint(self.l_size, self.h_size-1)
        img = F.resize(
            img, [size, size],
            interpolation=self.interpolation, antialias=self.antialias
        )
        pad_x = random.randint(0, self.end_size - size)
        pad_y = random.randint(0, self.end_size - size)
        img = F.pad(
            img, [
                pad_x, pad_y,
                self.end_size - size - pad_x, self.end_size - size - pad_y
            ]
        )
        return img