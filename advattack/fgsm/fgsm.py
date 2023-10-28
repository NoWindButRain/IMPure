import torch

from .. import Attack

class FGSM(Attack):
    def __init__(self, model, dataloader, save_path, criterion, eps=8):
        super(FGSM, self).__init__('FGSM_{}'.format(eps), model, dataloader, save_path)
        self.eps = eps / 255
        self.criterion = criterion


    def attack(self, images, labels=None):
        delta = torch.zeros_like(images, requires_grad=False).to(self.device)
        images.requires_grad = True
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        grad = torch.autograd.grad(loss, images,
            retain_graph=False, create_graph=False)[0]
        delta += self.eps * grad.sign()
        adv_images = (images + delta).detach()
        adv_images = torch.clamp(adv_images, min=0, max=1)
        delta = (adv_images - images).detach()
        return delta


