"""Restore immutable output from an earlier Kaggle version without rerunning completed training."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile
import base64
import io
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen
from urllib.parse import urlparse
from uuid import uuid4
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


def restore_download_manifest(manifest_path, staging):
    """Download historical output from a private manifest, checking every file's hash.

    Signed URLs stay in the private input dataset and are never included in logs or Git.
    The archive dataset provides a permanent recovery source once uploaded.
    """
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get('source') != 'thngonquang/deepweeds-day2' or manifest.get('version') != 3:
        raise ValueError('Unexpected recovery source')
    staging = Path(staging)
    entries = manifest['files']
    if len({entry['path'] for entry in entries}) != len(entries):
        raise ValueError('Duplicate recovery paths')
    for entry in entries:
        if not (staging / entry['path']).resolve().is_relative_to(staging.resolve()):
            raise ValueError('Unsafe recovery manifest path')
        url = urlparse(entry['url'])
        if url.scheme != 'https' or url.hostname not in ('www.kaggleusercontent.com', 'storage.googleapis.com'):
            raise ValueError('Unexpected recovery download host')
    if 'metadata_zip_b64' in manifest:
        contents = base64.b64decode(manifest['metadata_zip_b64'], validate=True)
        if hashlib.sha256(contents).hexdigest() != manifest['metadata_zip_sha256']:
            raise ValueError('Recovery metadata checksum mismatch')
        with zipfile.ZipFile(io.BytesIO(contents)) as archive:
            for entry in archive.infolist():
                if not (staging / entry.filename).resolve().is_relative_to(staging.resolve()):
                    raise ValueError('Unsafe recovery metadata path')
            for entry in archive.infolist():
                if entry.is_dir():
                    continue
                target = staging / entry.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                contents = archive.read(entry)
                if target.exists() and target.read_bytes() != contents:
                    raise ValueError(f'Refusing to overwrite different recovered data: {target}')
                if not target.exists():
                    with target.open('xb') as stream:
                        stream.write(contents)
    def fetch(entry):
        target = staging / entry['path']
        if target.exists():
            if target.stat().st_size != entry['size'] or file_sha256(target) != entry['sha256']:
                raise ValueError(f'Refusing to overwrite different recovered data: {target}')
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + '.partial.' + uuid4().hex)
        try:
            with urlopen(entry['url'], timeout=120) as response, partial.open('xb') as stream:
                shutil.copyfileobj(response, stream, length=4 << 20)
        except Exception:
            raise RuntimeError(f'Recovery download failed for {entry["path"]}; refresh private recovery links') from None
        if partial.stat().st_size != entry['size'] or file_sha256(partial) != entry['sha256']:
            raise ValueError(f'Recovery checksum mismatch: {entry["path"]}')
        if target.exists():
            raise ValueError(f'Refusing to replace existing recovery artifact: {target}')
        partial.rename(target)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, entries))
    print('Historical output downloads verified:', len(entries), 'files', flush=True)
    return staging


def restore_kaggle_input(destination, input_dir='/kaggle/input', required=True):
    candidates = [p.parent for p in Path(input_dir).rglob('training_results.json')
                  if (p.parent / 'backbone_results.json').is_file()]
    archives = list(Path(input_dir).rglob('lab_output_v3.zip'))
    manifests = list(Path(input_dir).rglob('recovery_downloads_v3.json'))
    if not candidates and not archives and len(manifests) == 1:
        staging = restore_download_manifest(manifests[0], Path(destination).parent / 'recovered_version3')
        candidates = [p.parent for p in staging.rglob('training_results.json')
                      if (p.parent / 'backbone_results.json').is_file()]
    if not candidates and len(archives) == 1:
        staging = Path(destination).parent / 'recovered_version3'
        with zipfile.ZipFile(archives[0]) as archive:
            # Check every path before creating files; preserve existing staging data.
            for entry in archive.infolist():
                target = staging / entry.filename
                if not target.resolve().is_relative_to(staging.resolve()):
                    raise ValueError(f'Unsafe recovery archive path: {entry.filename}')
            for entry in archive.infolist():
                if entry.is_dir():
                    continue
                target = staging / entry.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as source:
                    if target.exists():
                        digest = hashlib.sha256()
                        while block := source.read(1 << 20):
                            digest.update(block)
                        if file_sha256(target) != digest.hexdigest():
                            raise ValueError(f'Refusing to overwrite different recovered data: {target}')
                    else:
                        with target.open('xb') as output:
                            shutil.copyfileobj(source, output)
        candidates = [p.parent for p in staging.rglob('training_results.json')
                      if (p.parent / 'backbone_results.json').is_file()]
    if len(candidates) != 1:
        if not required and not candidates:
            return None
        raise RuntimeError(f'Expected one previous lab output, found {len(candidates)}; refusing full retraining')
    return restore_completed_training(candidates[0], destination)
