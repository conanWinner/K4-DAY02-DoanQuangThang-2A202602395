"""Losses and label-aware mixing; class weights are derived from training only."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def build_criterion(kind='ce', **kw):
    if kind == 'ce':
        return nn.CrossEntropyLoss()
    if kind == 'ls':
        return LabelSmoothingCE(kw.get('smoothing', 0.1))
    if kind == 'focal':
        return FocalLoss(kw.get('gamma', 2.0), kw.get('alpha'))
    if kind == 'ce_weighted':
        return nn.CrossEntropyLoss(weight=kw['weight'])
    raise ValueError(f'Unknown loss: {kind}')


class LabelSmoothingCE(nn.CrossEntropyLoss):
    def __init__(self, smoothing=0.1):
        super().__init__(label_smoothing=smoothing)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        if gamma < 0:
            raise ValueError('gamma must be nonnegative')
        self.gamma = gamma
        self.register_buffer('alpha', None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        log_pt = F.log_softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
        loss = -(1 - log_pt.exp()).pow(self.gamma) * log_pt
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        return loss.mean()


def class_weights(counts, beta=0.0):
    counts = torch.tensor(np.array(counts, copy=True), dtype=torch.float32)
    if counts.shape != (9,) or (counts <= 0).any() or not 0 <= beta < 1:
        raise ValueError('Need 9 positive train counts and beta in [0,1)')
    weights = counts.reciprocal() if beta == 0 else (1 - beta) / (1 - beta ** counts)
    return weights / weights.mean()


def mix_batch(x, y, alpha=1.0, mode='cutmix'):
    if alpha <= 0 or mode not in ('cutmix', 'mixup'):
        raise ValueError('alpha > 0 and mode in cutmix/mixup required')
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(len(x), device=x.device)
    if mode == 'mixup':
        mixed = lam * x + (1 - lam) * x[perm]
    else:
        h, w = x.shape[-2:]; ratio = np.sqrt(1 - lam)
        ch, cw = int(h * ratio), int(w * ratio)
        cy, cx = np.random.randint(h), np.random.randint(w)
        y1, y2 = max(0, cy - ch // 2), min(h, cy + (ch + 1) // 2)
        x1, x2 = max(0, cx - cw // 2), min(w, cx + (cw + 1) // 2)
        mixed = x.clone()
        mixed[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1 - (y2 - y1) * (x2 - x1) / (h * w)
    return mixed, (y, y[perm], lam)


def mixed_loss(criterion, logits, targets):
    a, b, lam = targets
    return lam * criterion(logits, a) + (1 - lam) * criterion(logits, b)
