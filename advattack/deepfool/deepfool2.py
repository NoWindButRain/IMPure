import numpy as np
import torch
import torch.nn as nn

from .. import Attack


class DeepFool2(Attack):
    def __init__(self, model, dataloader, save_path, criterion,
        eps=8, ieps=2, max_iter=50, overshoot=0.02):
        _name = 'DeepFool_{}_{}_{}'.format(eps, max_iter, overshoot)
        super(DeepFool2, self).__init__(_name, model, dataloader, save_path)
        self.criterion = criterion
        self.eps = eps / 255
        self.ieps = ieps / 255
        self.overshoot = overshoot
        self.max_iter = max_iter
        self.max_class_count = 10

    def attack(self, images, labels=None):
        adv_xs = []
        for i in range(len(images)):
            # print(i + 1, end=' ')
            adv_x = self._forward_indiv(
                images[i].unsqueeze(0), labels[i].unsqueeze(0), None)
            # distortion = torch.mean((adv_x - images[i].unsqueeze(0)) ** 2) / ((1 - 0) ** 2)
            # print(distortion.item())
            adv_xs.append(adv_x)
        adv_xs = torch.cat(adv_xs, 0)
        return (adv_xs - images).detach()

    def _forward_indiv(self, x, y, y_target=None):
        with torch.no_grad():
            logits = self.model(x)
            outputs = torch.argmax(logits, dim=1)
            if outputs != y:
                return x
        self.nb_classes = logits.size(-1)
        adv_x = x.clone().detach().requires_grad_()
        iteration = 0
        logits = self.model(adv_x)
        current = logits.max(1)[1].item()
        original = logits.max(1)[1].item()
        noise = torch.zeros(x.size()).to(self.device)
        w = torch.zeros(x.size()).to(self.device)
        w_norm = torch.zeros(x.size()).to(self.device)
        if self.nb_classes > self.max_class_count:
            labels = torch.argsort(logits[0], descending=True)
            labels = labels[: self.max_class_count]
        else:
            labels = torch.arange(self.nb_classes)
        while (current == original and iteration < self.max_iter):
            gradients_0 = torch.autograd.grad(
                logits[0, current], [adv_x], retain_graph=True)[0].detach()
            pert = np.inf
            for k in labels:
                if k == current:
                    continue
                gradients_1 = torch.autograd.grad(
                    logits[0, k], [adv_x], retain_graph=True)[0].detach()
                w_k = gradients_1 - gradients_0
                w_k_norm = torch.norm(w_k.flatten(1), 2, -1) + 1e-8
                f_k = logits[0, k] - logits[0, current]
                # if self.norm == np.inf:
                #     pert_k = (torch.abs(f_k) + 0.00001) / torch.norm(w_k.flatten(1), 1, -1)
                # elif self.norm == 2:
                pert_k = (torch.abs(f_k) + 1e-8) / w_k_norm
                if pert_k < pert:
                    pert = pert_k
                    w = w_k
                    w_norm = w_k_norm
            # if self.norm == np.inf:
            #     r_i = (pert + 1e-4) * w.sign()
            # elif self.norm == 2:
            r_i = (pert * w / w_norm).sign() * self.ieps
            # r_i = pert * w / w_norm
            noise += (1 + self.overshoot) * r_i
            noise = torch.clamp(noise, -self.eps, self.eps).detach()
            noise = noise.detach()
            adv_x = torch.clamp(noise + x, 0, 1).requires_grad_()
            logits = self.model(adv_x)
            current = logits.max(1)[1].item()
            iteration = iteration + 1
        # print(current, original)
        adv_x = torch.clamp(noise + x, 0, 1).detach()
        return adv_x

