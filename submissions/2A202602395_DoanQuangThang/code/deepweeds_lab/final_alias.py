"""Reuse one locked final evaluation under another reporting name, without inference."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import shutil


def copy_identical_or_new(source, target):
    source, target = Path(source), Path(target)
    if target.exists():
        def digest(path):
            value = hashlib.sha256()
            with path.open('rb') as stream:
                while block := stream.read(1 << 20):
                    value.update(block)
            return value.digest()
        if digest(source) != digest(target):
            raise ValueError(f'Refusing to replace different evaluation evidence: {target}')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def reuse_final_predictions(cfg, method, source_tag, target_tag, root):
    """Only alias a completed evaluation with the exact same config and inference method."""
    if source_tag == target_tag:
        raise ValueError('Final aliases require distinct reporting names')
    root = Path(root)
    source = root / 'final_cache' / source_tag / f'seed{cfg.seed}'
    target = root / 'final_cache' / target_tag / f'seed{cfg.seed}'
    locked = json.loads((source / 'locked_seed_config.json').read_text())
    result = json.loads((source / 'result.json').read_text())
    if locked['config'] != asdict(cfg) or locked['method'] != method:
        raise ValueError('Cannot reuse evaluation with a different config or method')
    if result['method'] != method or result['seed'] != cfg.seed or result['T'] != locked['T']:
        raise ValueError('Completed evaluation disagrees with its locked settings')
    if (target / 'result.json').exists():
        previous = json.loads((target / 'result.json').read_text())
        previous_lock = json.loads((target / 'locked_seed_config.json').read_text())
        if previous_lock != locked or any(previous[key] != result[key] for key in
                ('method', 'seed', 'T', 'macro_f1_val', 'macro_f1_test', 'top1_test', 'ece_test')):
            raise ValueError('Existing final result has different settings or predictions')
        for split in ('val', 'test'):
            src = root / 'predictions' / f'{source_tag}_seed{cfg.seed}_{split}.csv'
            dst = root / 'predictions' / f'{target_tag}_seed{cfg.seed}_{split}.csv'
            if src.read_bytes() != dst.read_bytes():
                raise ValueError(f'Existing final predictions differ: {dst}')
        return previous
    # Validate the entire copy plan first. Never replace older, independently measured output.
    pairs = [(root / 'predictions' / f'{source_tag}_seed{cfg.seed}_{split}.csv',
              root / 'predictions' / f'{target_tag}_seed{cfg.seed}_{split}.csv')
             for split in ('val', 'test')]
    pairs += [(source / name, target / name)
              for name in ('test_logits.npz', 'locked_seed_config.json')]
    pairs.append((root / 'curves' / f'{source_tag}_seed{cfg.seed}.png',
                  root / 'curves' / f'{target_tag}_seed{cfg.seed}.png'))
    for src, dst in pairs:
        if not src.is_file():
            raise FileNotFoundError(src)
        if dst.exists() and src.read_bytes() != dst.read_bytes():
            raise ValueError(f'Refusing to replace different evaluation evidence: {dst}')
    for src, dst in pairs:
        copy_identical_or_new(src, dst)
    result.update(exp_id=target_tag, reused_from=source_tag,
                  test_forward_reused=True, latency_reused=True)
    with (target / 'result.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print('FINAL TEST REUSED:', target_tag, 'from', source_tag, 'seed', cfg.seed, flush=True)
    return result
