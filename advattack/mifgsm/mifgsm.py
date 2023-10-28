import numpy as np
import torch
import torch.nn.functional as F

from .. import Attack

class MIFGSM(Attack):
    def __init__(self, model, dataloader, save_path, criterion,
            eps=8, steps=10, ieps=2, decay=1.0):
        _name = 'MIFGSM_{}_{}_{}'.format(eps, steps, ieps)
        super(MIFGSM, self).__init__(_name, model, dataloader, save_path)
        self.criterion = criterion
        self.eps = eps / 255
        self.steps = steps
        self.ieps = ieps / 255
        self.decay = decay


    def attack(self, images, labels=None):
        # random_start
        delta = torch.empty_like(images, requires_grad=False)
        delta = delta.uniform_(-self.eps, self.eps)
        delta = torch.clamp(images + delta, 0, 1) - images
        momentum = torch.zeros_like(images).detach().to(self.device)
        adv_images = images.clone().detach()
        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
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




