import os
from PIL import Image

from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms


class Attack:
    def __init__(self, name, model, dataloader, save_path,
            is_mask=False, mask_ratio=0.5, patch_size=16
        ):
        self.name = name
        self.model = model
        self.device = next(model.parameters()).device
        self.dataloader = dataloader
        self.batch_size = dataloader.batch_size
        self.dataset = dataloader.dataset
        # self.img_size = self.dataset.img_size
        # self.transform = dataloader.dataset.transform
        self.is_mask = is_mask
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.save_path = save_path


    def forward(self):
        correct = 0
        save_folder = '{}_{}'.format(self.model.name, self.name)
        if self.is_mask:
            save_folder += '_{}_{}'.format(self.mask_ratio, self.patch_size)
        save_path = os.path.join(self.save_path, save_folder)
        self.model = self.model.eval()
        for i, (images, labels) in enumerate(tqdm(self.dataloader)):
            # print("1:{}".format(torch.cuda.memory_allocated(5)))
            images, labels = images.to(self.device), labels.to(self.device)
            # print("2:{}".format(torch.cuda.memory_allocated(5)))
            delta = self.attack(images, labels)
            # print("3:{}".format(torch.cuda.memory_allocated(5)))
            if self.is_mask:
                delta, mask = self.random_masking(delta)
            adv_images = (images + delta).detach()
            with torch.no_grad():
                outputs = self.model(adv_images)
            preds = outputs.argmax(1)
            correct += (preds.cpu() == labels.cpu()).type(torch.float).sum().item()
            for j, adv_image in enumerate(adv_images):
                _index = i * self.batch_size + j
                _save_path = self.dataset.imgs[_index][0].replace(
                    self.dataset.root, save_path)
                self.saveimage(adv_image, _save_path)
            # if _index >= 127:
            #     break
            # torch.cuda.empty_cache()
        print(correct, len(self.dataset), correct / len(self.dataset))

    def attack(self, images, labels=None):
        raise NotImplementedError


    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = int((x.shape[-1]//3)**0.5)
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs


    def random_masking(self, images):
        N = images.shape[0]
        L = int((self.img_size / self.patch_size) ** 2)
        len_keep = int(L * (1 - self.mask_ratio))
        noise = torch.rand(N, L)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        mask = torch.ones([N, L])
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        mask = mask.unsqueeze(-1).repeat(1, 1, self.patch_size ** 2 * 3)
        mask = self.unpatchify(mask)
        im_masked = images * (1 - mask.to(self.device))
        return im_masked, mask


    def saveimage(self, image, save_path):
        _image = transform_invert(
            image.cpu().detach())
        os.makedirs(os.path.split(save_path)[0], exist_ok=True)
        # _image.save(save_path, quality=100, subsampling=0)
        _image.save(save_path+'.png')


    def image_clamp(self, images, bounds):
        for channel in range(3):
            _min, _max = bounds[channel]
            images[:, channel] = torch.clamp(images[:, channel], min=_min, max=_max)
        return images


def transform_invert(img_, transform_train=None):
    """
    将data 进行反transfrom操作
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    if 'Normalize' in str(transform_train):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_ = torch.clamp(img_  * 255, 0, 255)
    img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    img_ = np.around(np.array(img_))

    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]))

    return img_