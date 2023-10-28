import numpy as np
import torch
import torch.nn.functional as F

from .. import Attack

class DIFGSM(Attack):
    def __init__(self, model, dataloader, save_path, criterion,
            eps=8, steps=10, ieps=2, decay=1.0, resize_rate=0.9, diversity_prob=0.5):
        _name = 'DIFGSM_{}_{}_{}_{}'.format(eps, steps, ieps, decay)
        super(DIFGSM, self).__init__(_name, model, dataloader, save_path)
        self.criterion = criterion
        self.eps = eps / 255
        self.steps = steps
        self.ieps = ieps / 255
        self.decay = decay
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob


    def attack(self, images, labels=None):
        # random_start
        delta = torch.empty_like(images, requires_grad=False)
        delta = delta.uniform_(-self.eps, self.eps)
        delta = torch.clamp(images + delta, 0, 1) - images
        momentum = torch.zeros_like(images).detach().to(self.device)
        adv_images = images.clone().detach()
        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(self.input_diversity(adv_images))
            loss = self.criterion(outputs, labels)
            grad = torch.autograd.grad(loss, adv_images,
                retain_graph=False, create_graph=False)[0]
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad
            delta += self.ieps * grad.sign()
            delta = torch.clamp(delta, -self.eps, self.eps)
            adv_images = torch.clamp(images + delta, 0, 1)
            delta = (adv_images - images).detach()
        return (adv_images - images).detach()

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x


