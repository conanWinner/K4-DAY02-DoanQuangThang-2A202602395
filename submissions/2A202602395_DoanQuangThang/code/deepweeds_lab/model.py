"""timm models, frozen feature extraction and correctly grouped optimizer parameters."""
import copy
import torch
from torch import nn
import timm

SUGGESTED_BACKBONES = ['resnet50', 'resnext50_32x4d', 'convnext_tiny',
                       'deit_small_patch16_224', 'mobilenetv3_large_100']


def build_model(name, pretrained=True, num_classes=9, drop_rate=0.0, init='finetune'):
    if init not in ('scratch', 'frozen', 'finetune'):
        raise ValueError(f'Unknown initialization: {init}')
    kw = {'dynamic_img_size': True} if name.startswith(('vit_', 'deit_')) else {}
    model = timm.create_model(name, pretrained=pretrained and init != 'scratch',
                              num_classes=num_classes, drop_rate=drop_rate, **kw)
    if init == 'frozen':
        freeze_backbone(model)
    return model


def freeze_backbone(model):
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    for p in model.parameters():
        p.requires_grad_(id(p) in head_ids)
    model.eval()
    model.get_classifier().train()


def param_groups(model, lr_backbone, lr_head, weight_decay):
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    no_decay = model.no_weight_decay() if hasattr(model, 'no_weight_decay') else set()
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lr = lr_head if id(p) in head_ids else lr_backbone
        # Also exclude head bias and norm from decay, as required by GUIDE.
        wd = 0.0 if p.ndim <= 1 or name.endswith('.bias') or name in no_decay else weight_decay
        groups.setdefault((lr, wd), []).append(p)
    return [{'params': ps, 'lr': lr, 'weight_decay': wd} for (lr, wd), ps in groups.items()]


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size=224):
    from fvcore.nn import FlopCountAnalysis
    # fvcore counts one multiply-add as one operation. Unsupported ops are disclosed.
    m = copy.deepcopy(model).cpu().eval()
    # Expose attention matmuls so fvcore counts their MACs instead of skipping fused SDPA.
    for module in m.modules():
        if hasattr(module, 'fused_attn'):
            module.fused_attn = False
    analysis = FlopCountAnalysis(m, torch.zeros(1, 3, img_size, img_size))
    total = analysis.total()
    unsupported = dict(analysis.unsupported_ops())
    if unsupported:
        print('GMAC unsupported operations (not counted):', unsupported, flush=True)
    return total / 1e9
