"""One training implementation for screening, ablations and multi-seed finals."""
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import argparse
import copy
import json
import math
import random
import time
import numpy as np
import pandas as pd
import torch
from eval import compute_metrics, save_predictions
from . import dataset, model as models, losses
from .inference import apply_temperature


@dataclass
class Config:
    exp_id: str = 'T00'
    seed: int = 0
    fold: int = 0
    backbone: str = 'resnet50'
    init: str = 'finetune'
    drop_rate: float = 0.0
    img_size: int = 224
    aug: str = 'basic'
    sampler: str | None = None
    mix: str | None = None
    mix_alpha: float = 1.0
    loss: str = 'ce'
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    images_dir: str = 'data/images'
    labels_dir: str = 'data/labels'
    out_dir: str = 'runs'
    pred_dir: str = 'predictions'
    save_test_predictions: bool = False
    curves_dir: str = 'curves'
    device: str = 'cuda'


def run_dir(cfg):
    return Path(cfg.out_dir) / cfg.exp_id / f'seed{cfg.seed}'


def pred_path(cfg, split):
    return Path(cfg.pred_dir) / f'{cfg.exp_id}_seed{cfg.seed}_{split}.csv'


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg):
    return torch.optim.AdamW(models.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay))


def build_scheduler(optimizer, cfg, steps_per_epoch):
    total = cfg.epochs * steps_per_epoch
    warmup = max(1, round(cfg.warmup_epochs * steps_per_epoch))
    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1, (step - warmup) / max(1, total - warmup))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class EMA:
    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        source = model.state_dict()
        parameters = dict(self.model.named_parameters())
        for name, value in self.model.state_dict().items():
            if name in parameters:
                value.mul_(self.decay).add_(source[name], alpha=1 - self.decay)
            else:
                # BN statistics track the current training distribution, integer counters copied.
                value.copy_(source[name])


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg, device, ema=None):
    model.train()
    if cfg.init == 'frozen':
        model.eval(); model.get_classifier().train()
    total, count = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        targets = None
        if cfg.mix:
            x, targets = losses.mix_batch(x, y, cfg.mix_alpha, cfg.mix)
        with torch.autocast(device_type=torch.device(device).type,
                            enabled=cfg.amp and torch.device(device).type == 'cuda'):
            logits = model(x)
            loss = losses.mixed_loss(criterion, logits, targets) if targets is not None else criterion(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss; experiment stopped')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer); scaler.update()
        if scaler.get_scale() >= previous_scale:
            scheduler.step()
            if ema:
                ema.update(model)
        total += float(loss.detach()) * len(y); count += len(y)
    return {'train_loss': total / count, 'lr': optimizer.param_groups[0]['lr']}


def evaluate(model, loader, criterion, device):
    model.eval(); names, ys, zs, total, count = [], [], [], 0.0, 0
    with torch.inference_mode():
        for x, y, filenames in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            total += float(criterion(logits, y)) * len(y); count += len(y)
            names.extend(filenames); ys.extend(y.cpu().tolist()); zs.append(logits.float().cpu().numpy())
    return names, np.asarray(ys, dtype=np.int64), np.concatenate(zs), total / count


def plot_curves(history, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    h = pd.DataFrame(history); fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(h.epoch, h.train_loss, label='train'); axes[0].plot(h.epoch, h.val_loss, label='val')
    axes[0].set_ylabel('Loss'); axes[0].legend()
    axes[1].plot(h.epoch, h.macro_f1, label='val macro-F1'); axes[1].plot(h.epoch, h.top1, label='val top-1')
    axes[1].legend(); axes[1].set_ylabel('Score')
    axes[2].plot(h.epoch, h.lr); axes[2].set_ylabel('Learning rate (epoch end)')
    for ax in axes:
        ax.set_xlabel('Epoch'); ax.grid(alpha=0.2)
    fig.suptitle(title); fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2), encoding='utf-8')


def load_checkpoint_model(cfg, device=None):
    m = models.build_model(cfg.backbone, pretrained=False, init='scratch', drop_rate=cfg.drop_rate)
    m.load_state_dict(torch.load(run_dir(cfg) / 'best.pt', map_location='cpu', weights_only=False)['model'])
    return m.to(device or cfg.device).eval()


def run(cfg):
    if cfg.epochs < 1 or cfg.batch_size < 2:
        raise ValueError('Positive epoch count and batch size >=2 required')
    if cfg.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('GPU required for full experiments; enable Kaggle GPU')
    folder = run_dir(cfg); folder.mkdir(parents=True, exist_ok=True)
    config_path, result_path = folder / 'config.json', folder / 'result.json'
    if config_path.exists() and json.loads(config_path.read_text()) != asdict(cfg):
        raise ValueError(f'Refusing to overwrite a different experiment: {folder}')
    if result_path.exists():
        print('Reuse completed run:', folder, flush=True)
        return json.loads(result_path.read_text())
    set_seed(cfg.seed); write_json(config_path, asdict(cfg))
    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, cfg.fold)
    split_check = dataset.check_split(train_df, val_df, test_df, cfg.images_dir)
    write_json(folder / 'split_checks.json', split_check)
    model = models.build_model(cfg.backbone, init=cfg.init, drop_rate=cfg.drop_rate).to(cfg.device)
    pcfg = model.pretrained_cfg
    mean, std = pcfg.get('mean', dataset.IMAGENET_MEAN), pcfg.get('std', dataset.IMAGENET_STD)
    write_json(folder / 'pretrained.json', pcfg)
    transform_train = dataset.build_transforms(True, cfg.img_size, cfg.aug, mean, std)
    transform_val = dataset.build_transforms(False, cfg.img_size, mean=mean, std=std)
    train_loader = dataset.make_loader(train_df, cfg.images_dir, transform_train, cfg.batch_size,
                                      True, cfg.sampler, cfg.num_workers, cfg.seed)
    val_loader = dataset.make_loader(val_df, cfg.images_dir, transform_val, cfg.batch_size,
                                    False, num_workers=cfg.num_workers, seed=cfg.seed)
    counts = train_df.Label.value_counts().reindex(range(9), fill_value=0).to_numpy()
    criterion = losses.build_criterion(cfg.loss, smoothing=cfg.label_smoothing,
            gamma=cfg.focal_gamma, weight=losses.class_weights(counts, cfg.class_weight_beta or 0)).to(cfg.device)
    # Validation loss always CE, comparable across focal/weighted/mixed objectives.
    val_criterion = torch.nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg); scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp and cfg.device.startswith('cuda'))
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay is not None else None
    history, best_f1, best_epoch, first = [], -1.0, 0, 1
    last_path = folder / 'last.pt'
    if last_path.exists():
        state = torch.load(last_path, map_location=cfg.device, weights_only=False)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler'])
        if ema:
            ema.model.load_state_dict(state['ema'])
        history, best_f1, best_epoch, first = state['history'], state['best_f1'], state['best_epoch'], state['epoch'] + 1
        random.setstate(state['random']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'].cpu())
        train_loader.generator.set_state(state['loader_rng'].cpu())
        if torch.cuda.is_available() and state['cuda'] is not None:
            torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])
        print('Resume epoch:', first, flush=True)
    for epoch in range(first, cfg.epochs + 1):
        start = time.perf_counter()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, cfg.device, ema)
        _, y, z, val_loss = evaluate(ema.model if ema else model, val_loader, val_criterion, cfg.device)
        p = apply_temperature(z, 1); metric = compute_metrics(y, p.argmax(1), p)
        row = dict(epoch=epoch, **tr, val_loss=val_loss, macro_f1=metric['macro_f1'],
                   top1=metric['top1'], seconds=time.perf_counter() - start)
        history.append(row)
        if metric['macro_f1'] > best_f1:  # strict > keeps earlier epoch on ties
            best_f1, best_epoch = metric['macro_f1'], epoch
            torch.save({'model': (ema.model if ema else model).state_dict(), 'epoch': epoch}, folder / 'best.pt')
        pd.DataFrame(history).to_csv(folder / 'history.csv', index=False)
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
            ema=ema.model.state_dict() if ema else None, epoch=epoch, history=history,
            best_f1=best_f1, best_epoch=best_epoch, random=random.getstate(), numpy=np.random.get_state(),
            torch=torch.get_rng_state(), loader_rng=train_loader.generator.get_state(),
            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None), last_path)
        print(cfg.exp_id, f'seed={cfg.seed}', row, flush=True)
        plot_curves(history, Path(cfg.curves_dir) / f'{cfg.exp_id}_seed{cfg.seed}.png', f'{cfg.exp_id} seed{cfg.seed} {cfg.backbone}')
    model.load_state_dict(torch.load(folder / 'best.pt', map_location=cfg.device, weights_only=False)['model'])
    names, y, z, _ = evaluate(model, val_loader, val_criterion, cfg.device)
    np.savez(folder / 'val_logits.npz', filenames=names, labels=y, logits=z)
    p = apply_temperature(z, 1); save_predictions(pred_path(cfg, 'val'), names, y, p)
    metrics = compute_metrics(y, p.argmax(1), p)
    result = dict(exp_id=cfg.exp_id, seed=cfg.seed, backbone=cfg.backbone, init=cfg.init,
                  pretrained_tag=pcfg.get('tag', ''), pretrained_architecture=pcfg.get('architecture', cfg.backbone),
                  params_m=models.count_params(model), gmac=models.count_gmacs(model, cfg.img_size),
                  gmac_tool='fvcore (unsupported operations printed)', best_epoch=best_epoch,
                  epochs=cfg.epochs, img_size=cfg.img_size, train_seconds_per_epoch=np.mean([r['seconds'] for r in history]),
                  **{f'{k}_val': metrics[k] for k in ('macro_f1', 'top1', 'balanced_acc', 'ece')})
    if cfg.save_test_predictions:
        # Reuse saved raw predictions after interruption instead of forwarding test again.
        archive = folder / 'test_logits.npz'
        if archive.exists():
            a = np.load(archive); names, y, z = a['filenames'].tolist(), a['labels'], a['logits']
        else:
            test_loader = dataset.make_loader(test_df, cfg.images_dir, transform_val, cfg.batch_size,
                    False, num_workers=cfg.num_workers)
            names, y, z, _ = evaluate(model, test_loader, val_criterion, cfg.device)
            np.savez(archive, filenames=names, labels=y, logits=z)
        p = apply_temperature(z, 1); save_predictions(pred_path(cfg, 'test'), names, y, p)
        result.update({f'{k}_test': v for k, v in compute_metrics(y, p.argmax(1), p).items()})
    write_json(result_path, result)
    return jsonable(result)


def parse_overrides(pairs):
    known = {f.name: f for f in fields(Config)}; result = {}
    defaults = asdict(Config())
    for pair in pairs:
        key, value = pair.split('=', 1)
        if key not in known:
            raise ValueError(f'Unknown Config field: {key}')
        default = defaults[key]
        if isinstance(default, bool):
            if value.lower() not in ('true', 'false'):
                raise ValueError(f'{key} must be true/false')
            parsed = value.lower() == 'true'
        elif value.lower() == 'none':
            if 'None' not in str(known[key].type):
                raise ValueError(f'{key} is not nullable')
            parsed = None
        elif default is None:
            parsed = float(value) if key in ('ema_decay', 'class_weight_beta') else value
        else:
            parsed = type(default)(value)
        result[key] = parsed
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--set', nargs='*', default=[])
    print(json.dumps(run(Config(**parse_overrides(parser.parse_args().set))), indent=2))


if __name__ == '__main__':
    main()
