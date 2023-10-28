import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .. import Attack


class CW2(Attack):
    def __init__(self, model, dataloader, save_path, criterion,
        eps=8, initial_const=1e-2, confidence=0,
        binary_search_steps=5, max_iterations=1000, lr=5e-3):
        _name = 'CW2_{}_{}_{}_{}'.format(eps, binary_search_steps, confidence, lr)
        super(CW2, self).__init__(_name, model, dataloader, save_path)
        self.criterion = criterion
        self.eps = eps / 255
        self.initial_const = initial_const
        self.confidence = confidence
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.lr = lr
        self.n_classes = 1000

    def attack(self, images, labels=None):
        INF = float("inf")
        x = images
        y = labels
        lower_bound = [0.0] * len(x)
        upper_bound = [1e10] * len(x)
        const = x.new_ones(len(x), 1) * self.initial_const
        o_bestl2 = [INF] * len(x)
        o_bestscore = [-1.0] * len(x)
        x = torch.clamp(x, 0, 1)
        ox = x.clone().detach()  # save the original x
        o_bestattack = x.clone().detach()
        x = (x - 0) / (1 - 0)
        x = torch.clamp(x, 0, 1)
        x = x * 2 - 1
        x = torch.arctanh(x * 0.999999)
        modifier = torch.zeros_like(x, requires_grad=True)
        y_onehot = torch.nn.functional.one_hot(y, self.n_classes).to(torch.float)
        f_fn = lambda real, other, targeted=False: torch.max(
            ((other - real) if targeted else (real - other)) + self.confidence,
            torch.tensor(0.0).to(real.device),
        )
        l2dist_fn = lambda x, y: torch.pow(x - y, 2).sum(list(range(len(x.size())))[1:])
        optimizer = torch.optim.Adam([modifier], lr=self.lr)
        for outer_step in range(self.binary_search_steps):
            # Initialize some values needed for the inner loop
            bestl2 = [INF] * len(x)
            bestscore = [-1.0] * len(x)
            # Inner loop performing attack iterations
            for i in range(self.max_iterations):
                # One attack step
                new_x = (torch.tanh(modifier + x) + 1) / 2
                new_x = new_x * (1 - 0) + 0
                delta = new_x - images
                delta = torch.clamp(delta, -self.eps, self.eps)
                nex_x = images + delta
                logits = self.model(nex_x)
                real = torch.sum(y_onehot * logits, 1)
                other, _ = torch.max((1 - y_onehot) * logits - y_onehot * 1e4, 1)
                optimizer.zero_grad()
                f = f_fn(real, other)
                l2 = l2dist_fn(new_x, ox)
                loss = (const * f + l2).sum()
                loss.backward()
                optimizer.step()
                # Update best results
                for n, (l2_n, logits_n, new_x_n) in enumerate(zip(l2, logits, new_x)):
                    y_n = y[n]
                    succeeded = self.compare(logits_n, y_n, is_logits=True)
                    if l2_n < o_bestl2[n] and succeeded:
                        pred_n = torch.argmax(logits_n)
                        o_bestl2[n] = l2_n
                        o_bestscore[n] = pred_n
                        o_bestattack[n] = new_x_n
                        # l2_n < o_bestl2[n] implies l2_n < bestl2[n] so we modify inner loop variables too
                        bestl2[n] = l2_n
                        bestscore[n] = pred_n
                    elif l2_n < bestl2[n] and succeeded:
                        bestl2[n] = l2_n
                        bestscore[n] = torch.argmax(logits_n)
            # Binary search step
            for n in range(len(x)):
                y_n = y[n]
                if self.compare(bestscore[n], y_n) and bestscore[n] != -1:
                    # Success, divide const by two
                    upper_bound[n] = min(upper_bound[n], const[n])
                    if upper_bound[n] < 1e9:
                        const[n] = (lower_bound[n] + upper_bound[n]) / 2
                else:
                    # Failure, either multiply by 10 if no solution found yet
                    # or do binary search with the known upper bound
                    lower_bound[n] = max(lower_bound[n], const[n])
                    if upper_bound[n] < 1e9:
                        const[n] = (lower_bound[n] + upper_bound[n]) / 2
                    else:
                        const[n] *= 10
            new_x = (torch.tanh(modifier.detach() + x) + 1) / 2
            new_x = new_x * (1 - 0) + 0
            delta = new_x - images
            delta = torch.clamp(delta, -self.eps, self.eps)
            nex_x = images + delta
            modifier = (torch.arctanh(nex_x * 2 - 1) - x).requires_grad_()
            optimizer = torch.optim.Adam([modifier], lr=self.lr)
        return (o_bestattack - images).detach()

    def compare(self, pred, label, is_logits=False, targeted=False):
        if is_logits:
            pred_copy = pred.clone().detach()
            pred_copy[label] += -self.confidence if targeted else self.confidence
            pred = torch.argmax(pred_copy)
        return pred == label if targeted else pred != label
