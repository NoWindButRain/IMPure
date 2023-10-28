import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .. import Attack


class CW(Attack):
    def __init__(self, model, dataloader, save_path, criterion,
        eps=8, c=1, kappa=0, steps=50, lr=0.01):
        _name = 'CW_{}_{}_{}_{}'.format(eps, steps, kappa, lr)
        super(CW, self).__init__(_name, model, dataloader, save_path)
        self.criterion = criterion
        self.eps = eps / 255
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr


    def attack(self, images, labels=None):
        x = images.clone().detach().to(self.device)
        y = labels.clone().detach().to(self.device)
        w = self.inverse_tanh_space(x).detach()
        w.requires_grad = True
        best_adv_images = x.clone().detach()
        best_L2 = 1e10 * torch.ones((len(x))).to(self.device)
        prev_cost = 1e10
        dim = len(x.shape)
        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()
        optimizer = optim.Adam([w], lr=self.lr)
        for step in range(self.steps):
            # Get adversarial images
            adv_images = self.tanh_space(w)
            # Calculate loss
            current_L2 = MSELoss(Flatten(adv_images),
                Flatten(x)).sum(dim=1)
            L2_loss = current_L2.sum()
            outputs = self.model(adv_images)
            f_loss = self.f(outputs, y).sum()
            cost = L2_loss + self.c * f_loss
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            w.requires_grad = False
            delta = torch.clamp(
                (self.tanh_space(w) - images).detach(), -self.eps, self.eps)
            _image = self.inverse_tanh_space(images + delta)
            w[:, :, :, :] = _image[:, :, :, :]
            w.requires_grad = True
            # Update adversarial images
            _, pre = torch.max(outputs.detach(), 1)
            correct = (pre == y).float()
            # filter out images that get either correct predictions or non-decreasing loss,
            # i.e., only images that are both misclassified and loss-decreasing are left
            mask = (1 - correct) * (best_L2 > current_L2.detach())
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2
            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images

            # Early stop when loss does not converge.
            # max(.,1) To prevent MODULO BY ZERO error in the next step.
            if step % max(self.steps // 10, 1) == 0:
                if cost.item() > prev_cost:
                    return (best_adv_images - images).detach()
                prev_cost = cost.item()
        return (best_adv_images - images).detach()


    def tanh_space(self, x):
        return 1 / 2 * (torch.tanh(x) + 1)


    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        return self.atanh(x * 2 - 1)


    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))


    # f-function in the paper
    def f(self, outputs, labels):
        one_hot_labels = torch.eye(len(outputs[0])).to(self.device)[labels]
        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)  # get the second largest logit
        j = torch.masked_select(outputs, one_hot_labels.bool())  # get the largest logit
        return torch.clamp((j - i), min=-self.kappa)

