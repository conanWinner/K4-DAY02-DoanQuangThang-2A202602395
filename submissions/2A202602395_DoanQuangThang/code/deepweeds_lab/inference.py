"""Deterministic inference, TTA and validation-fitted calibration."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import minimize_scalar


def predict_logits(model, loader, device, view=None):
    model.eval(); names, labels, outputs = [], [], []
    with torch.inference_mode():
        for x, y, filenames in loader:
            x = x.to(device, non_blocking=True)
            z = model(view(x) if view else x)
            names.extend(filenames); labels.extend(y.tolist()); outputs.append(z.float().cpu().numpy())
    return names, np.asarray(labels, dtype=np.int64), np.concatenate(outputs)


def view_identity(x):
    return x


def view_hflip(x):
    return x.flip(-1)


def views_multicrop(x, crop):
    h, w = x.shape[-2:]
    if crop > min(h, w) or crop <= 0:
        raise ValueError('Crop must fit inside input')
    return [x[..., y:y + crop, z:z + crop] for y, z in
            [(0, 0), (0, w - crop), (h - crop, 0), (h - crop, w - crop),
             ((h - crop) // 2, (w - crop) // 2)]]


def views_multiscale(x, sizes):
    return [F.interpolate(x, size=(s, s), mode='bilinear', align_corners=False) for s in sizes]


def apply_temperature(logits, T):
    if not np.isfinite(T) or T <= 0:
        raise ValueError('Temperature must be finite and positive')
    z = np.asarray(logits, dtype=np.float64) / T
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


def aggregate_views(logits_per_view, space='prob'):
    zs = np.stack(logits_per_view)
    if space == 'logit':
        return apply_temperature(zs.mean(0), 1)
    if space == 'prob':
        return np.mean([apply_temperature(z, 1) for z in zs], axis=0)
    raise ValueError('space must be prob or logit')


def ensemble_probs(list_of_probs):
    p = np.stack(list_of_probs).mean(0)
    if (p < 0).any() or not np.isfinite(p).all():
        raise ValueError('Invalid probabilities')
    return p / p.sum(1, keepdims=True)


def fit_temperature(val_logits, val_labels):
    logits = np.asarray(val_logits, dtype=np.float64)
    labels = np.asarray(val_labels, dtype=np.int64)
    def objective(log_T):
        p = apply_temperature(logits, np.exp(log_T))
        return -np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1)).mean()
    fit = minimize_scalar(objective, bounds=(-4, 4), method='bounded')
    # Include T=1 to prevent numerical optimization making validation NLL worse.
    return float(np.exp(fit.x)) if fit.success and fit.fun < objective(0) else 1.0


def fuse_conv_bn(model):
    result = copy.deepcopy(model).eval()
    def fuse(module):
        for child in module.children():
            fuse(child)
        # Sequential guarantees adjacency; named timm Conv->BN is restricted to standard pairs.
        pairs = []
        if isinstance(module, nn.Sequential):
            names = list(module._modules)
            pairs += list(zip(names, names[1:]))
        for names in [('conv1', 'bn1'), ('conv2', 'bn2'), ('conv3', 'bn3'),
                      ('conv_dw', 'bn1'), ('conv_pw', 'bn2'), ('conv_pwl', 'bn3'),
                      ('conv_stem', 'bn1')]:
            # MobileNet blocks have architecture-dependent ordering: only conventional ResNet
            # pairs are fused outside Sequential. Unsupported pairs are left intact.
            if names[0].startswith('conv') and names[0] in ('conv1', 'conv2', 'conv3'):
                pairs.append(names)
        for a, b in pairs:
            conv, bn = getattr(module, a, None), getattr(module, b, None)
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                setattr(module, a, torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn))
                setattr(module, b, nn.Identity())
    fuse(result)
    return result


def strategy_logits(model, x, method='single', image_size=224):
    """All views are generated deterministically from already normalized inputs."""
    if method in ('single', 'temperature', 'amp', 'fused'):
        return [model(x)]
    if method in ('hflip_prob', 'hflip_logit'):
        return [model(x), model(view_hflip(x))]
    if method == 'fivecrop':
        return [model(v) for v in views_multicrop(x, image_size)]
    if method == 'resolution256':
        return [model(x)]  # loader performs deterministic 256-pixel preprocessing
    raise ValueError(f'Unknown strategy: {method}')


def predict_strategy(model, loader, device, method='single', image_size=224, T=1.0):
    model.eval(); names, ys, effective_logits = [], [], []
    with torch.inference_mode():
        for x, y, filenames in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=torch.device(device).type,
                                enabled=method == 'amp' and torch.device(device).type == 'cuda'):
                zs = strategy_logits(model, x, method, image_size)
            zs = [z.float().cpu().numpy() for z in zs]
            if method == 'hflip_logit' or len(zs) == 1:
                z = np.mean(zs, axis=0)
            else:
                # log(mean probabilities) is an equivalent logit; calibrate the aggregate.
                z = np.log(np.clip(aggregate_views(zs), 1e-12, 1))
            effective_logits.append(z); names.extend(filenames); ys.extend(y.tolist())
    z = np.concatenate(effective_logits)
    return names, np.asarray(ys, dtype=np.int64), z, apply_temperature(z, T)
