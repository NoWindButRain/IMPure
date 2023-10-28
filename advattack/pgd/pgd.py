import numpy as np
import torch

from .. import Attack

class PGD(Attack):
    def __init__(self, model, dataloader, save_path, criterion, eps=8, steps=10, ieps=2):
        _name = 'PGD_{}_{}_{}'.format(eps, steps, ieps)
        super(PGD, self).__init__(_name, model, dataloader, save_path)
        self.criterion = criterion
        self.eps = eps / 255
        self.steps = steps
        self.ieps = ieps / 255


    def attack(self, images, labels=None):
        # random_start
        delta = torch.empty_like(images, requires_grad=False)
        delta = delta.uniform_(-self.eps, self.eps)
        delta = torch.clamp(images + delta, 0, 1) - images
        adv_images = images.clone().detach()
        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            loss = self.criterion(outputs, labels)
            grad = torch.autograd.grad(loss, adv_images,
                retain_graph=False, create_graph=False)[0]
            delta += self.ieps * grad.sign()
            delta = torch.clamp(delta, -self.eps, self.eps)
            adv_images = torch.clamp(images + delta, 0, 1)
            delta = (adv_images - images).detach()
        return (adv_images - images).detach()


