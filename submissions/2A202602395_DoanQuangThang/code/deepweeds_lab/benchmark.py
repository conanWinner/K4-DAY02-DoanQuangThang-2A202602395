"""Synchronized wall-clock timings, excluding disk I/O and preprocessing."""
import contextlib
import copy
import time
import numpy as np
import torch
from .inference import strategy_logits


def bench(fn, warmup=10, iters=100, sync=None):
    if warmup < 10 or iters < 50:
        raise ValueError('At least 10 warmup and 50 measured iterations required')
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    sync(); values = []
    for _ in range(iters):
        sync(); start = time.perf_counter(); fn(); sync()
        values.append((time.perf_counter() - start) * 1000)
    return dict(zip(('p50', 'p95', 'p99'), map(float, np.percentile(values, [50, 95, 99]))),
                mean=float(np.mean(values)), n=iters, warmup=warmup)


def strategy_latency(model, batch_size=1, img_size=224, method='single', dtype='fp32',
                     device='cuda', warmup=10, iters=100, temperature=1.0):
    device = torch.device(device)
    # Preserve the model's original dtype; FP16 measurements use an independent copy.
    m = copy.deepcopy(model).to(device).eval() if dtype == 'fp16' else model.to(device).eval()
    if dtype == 'fp16':
        m.half()
    input_size = round(img_size * 256 / 224) if method == 'fivecrop' else (256 if method == 'resolution256' else img_size)
    x = torch.randn(batch_size, 3, input_size, input_size, device=device,
                    dtype=torch.float16 if dtype == 'fp16' else torch.float32)
    def forward():
        with torch.inference_mode(), torch.autocast(device_type=device.type,
                enabled=(dtype == 'amp' or method == 'amp') and device.type == 'cuda'):
            zs = strategy_logits(m, x, method, img_size)
            if len(zs) > 1:
                if method == 'hflip_logit':
                    p = (torch.stack(zs).mean(0) / temperature).softmax(-1)
                else:
                    p = torch.stack([z.softmax(-1) for z in zs]).mean(0)
                    if temperature != 1:
                        p = p.clamp_min(1e-12).pow(1 / temperature)
                        p = p / p.sum(-1, keepdim=True)
            else:
                p = (zs[0] / temperature).softmax(-1)
            return p
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == 'cuda' else None
    result = bench(forward, warmup, iters, sync)
    result.update(gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                  dtype='amp' if method == 'amp' else dtype, batch=batch_size, img_size=input_size,
                  images_per_s=batch_size * 1000 / result['p50'], torch=torch.__version__,
                  preprocessing=False, method=method, temperature=temperature, fused_bn=method == 'fused')
    return result


def latency_report(model, batch_size, img_size, dtype='fp32', device='cuda', warmup=10, iters=100):
    return strategy_latency(model, batch_size, img_size, dtype=dtype, device=device,
                            warmup=warmup, iters=iters)


def tta_latency(model, k_views, **kw):
    if k_views not in (2, 5):
        raise ValueError('Supported actual TTA measurements: 2 flip views or 5 crops')
    return strategy_latency(model, method='hflip_prob' if k_views == 2 else 'fivecrop', **kw)
