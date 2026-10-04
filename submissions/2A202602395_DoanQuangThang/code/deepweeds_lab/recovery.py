"""Restore immutable output from an earlier Kaggle version without rerunning completed training."""
from pathlib import Path
import hashlib
import json
import shutil
from .train import write_json


def restore_completed_training(source, destination):
    source, destination = Path(source), Path(destination)
    groups = {}
    for filename, expected_ids in (
            ('backbone_results.json', {f'B{i:02d}' for i in range(1, 6)}),
            ('training_results.json', {f'T{i:02d}' for i in range(10)})):
        rows = json.loads((source / filename).read_text())
        if {row['exp_id'] for row in rows} != expected_ids or len(rows) != len(expected_ids):
            raise ValueError(f'Incomplete training evidence: {filename}')
        groups[filename] = rows
        for row in rows:
            folder = source / 'runs' / row['exp_id'] / f"seed{row['seed']}"
            for name in ('best.pt', 'config.json', 'result.json', 'history.csv', 'pretrained.json', 'val_logits.npz'):
                if not (folder / name).is_file():
                    raise FileNotFoundError(f'Missing recovery artifact: {folder / name}')
            config = json.loads((folder / 'config.json').read_text())
            if config != row['config']:
                raise ValueError(f'Recovery config disagrees with recorded result: {folder}')
    destination.mkdir(parents=True, exist_ok=True)
    copied, reused = 0, 0
    # Preserve every old file, including checkpoints and error evidence. A prior error is
    # archived separately so it cannot be mistaken for the status of the new execution.
    for item in source.rglob('*'):
        if not item.is_file():
            continue
        relative = item.relative_to(source)
        if relative == Path('execution_error.json'):
            relative = Path('prior_errors') / 'version3_execution_error.json'
        elif relative == Path('environment.json'):
            relative = Path('prior_environment') / 'version3_environment.json'
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.stat().st_size != item.stat().st_size:
                raise ValueError(f'Refusing to overwrite different recovered data: {target}')
            # Identical-size files also require matching bytes to avoid accidental replacement.
            if file_sha256(target) != file_sha256(item):
                raise ValueError(f'Refusing to overwrite different recovered data: {target}')
            reused += 1
        else:
            shutil.copy2(item, target); copied += 1
    record = {'source': str(source), 'completed_backbones': 5, 'completed_training_configs': 10,
              'copied_files': copied, 'reused_files': reused, 'policy': 'reuse completed runs; test selection unchanged'}
    write_json(destination / 'recovery_manifest.json', record)
    print('RECOVERY VERIFIED:', record, flush=True)
    return record


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def restore_kaggle_input(destination, input_dir='/kaggle/input', required=True):
    candidates = [p.parent for p in Path(input_dir).rglob('training_results.json')
                  if (p.parent / 'backbone_results.json').is_file()]
    if len(candidates) != 1:
        if not required and not candidates:
            return None
        raise RuntimeError(f'Expected one previous lab output, found {len(candidates)}; refusing full retraining')
    return restore_completed_training(candidates[0], destination)
