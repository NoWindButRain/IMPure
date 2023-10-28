import numpy as np
import torch
import torch.nn as nn

from .. import Attack


class DeepFool(Attack):
    def __init__(self, model, dataloader, save_path, criterion,
        steps=50, overshoot=0.02):
        _name = 'DeepFool_{}_{}'.format(steps, overshoot)
        super(DeepFool, self).__init__(_name, model, dataloader, save_path)
        self.steps = steps
        self.overshoot = overshoot
        self.max_class_count = 10


    def attack(self, images, labels=None):
        x = images.clone().detach().to(self.device)
        y = labels.clone().detach().to(self.device)
        batch_size = len(x)
        correct = torch.tensor([True] * batch_size)
        target_labels = y.clone().detach().to(self.device)
        curr_steps = 0
        adv_images = []
        for idx in range(batch_size):
            image = x[idx:idx + 1].clone().detach()
            adv_images.append(image)
        while (True in correct) and (curr_steps < self.steps):
            print(curr_steps)
            for idx in range(batch_size):
                print(idx)
                if not correct[idx]: continue
                early_stop, pre, adv_image = self._forward_indiv(adv_images[idx], y[idx])
                adv_images[idx] = adv_image
                target_labels[idx] = pre
                if early_stop:
                    correct[idx] = False
            curr_steps += 1
        adv_images = torch.cat(adv_images).detach()
        return (adv_images - images).detach()


    def _forward_indiv(self, image, label):
        image.requires_grad = True
        fs = self.model(image)[0]
        _, pre = torch.max(fs, dim=0)
        if pre != label:
            return (True, pre, image)
        if len(fs) > self.max_class_count:
            labels = torch.argsort(fs)[::-1]
            labels = labels[: self.max_class_count]
        else:
            labels = torch.arange(len(fs))
        ws = self._construct_jacobian(fs[labels], image)
        image = image.detach()
        f_0 = fs[label]
        w_0 = ws[label]
        wrong_classes = [i for i in range(len(fs)) if i != label]
        f_k = fs[wrong_classes]
        w_k = ws[wrong_classes]
        f_prime = f_k - f_0
        w_prime = w_k - w_0
        value = torch.abs(f_prime) \
                / torch.norm(nn.Flatten()(w_prime), p=2, dim=1)
        _, hat_L = torch.min(value, 0)
        delta = (torch.abs(f_prime[hat_L]) * w_prime[hat_L] \
                 / (torch.norm(w_prime[hat_L], p=2) ** 2))
        target_label = hat_L if hat_L < label else hat_L + 1
        adv_image = image + (1 + self.overshoot) * delta
        adv_image = torch.clamp(adv_image, min=0, max=1).detach()
        return (False, target_label, adv_image)

    # https://stackoverflow.com/questions/63096122/pytorch-is-it-possible-to-differentiate-a-matrix
    # torch.autograd.functional.jacobian is only for torch >= 1.5.1
    def _construct_jacobian(self, y, x):
        x_grads = []
        for idx, y_element in enumerate(y):
            if x.grad is not None:
                x.grad.zero_()
            y_element.backward(retain_graph=(False or idx + 1 < len(y)))
            x_grads.append(x.grad.clone().detach())
        return torch.stack(x_grads).reshape(*y.shape, *x.shape)














