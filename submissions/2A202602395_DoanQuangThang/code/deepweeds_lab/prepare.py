"""Download original files, verify checksum and visualize actual training data."""
from pathlib import Path
import hashlib
import json
import platform
import time
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import torch
from PIL import Image
from . import dataset
from .train import set_seed, write_json

MD5 = 'b7b30f96d466fba86016aa5a26606e0f'


def download(url, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    # Write into a new partial file; never silently replace an existing download.
    partial = path.with_name(path.name + f'.partial.{time.time_ns()}')
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'DeepWeeds-Lab/1.0'})
            with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as output:
                while chunk := response.read(1 << 20):
                    output.write(chunk)
            partial.rename(path)
            print('Downloaded:', path, flush=True)
            return
        except Exception:
            if attempt == 3:
                raise
            # Preserve failed partial data and start a distinct file on retry.
            partial = path.with_name(path.name + f'.partial.{time.time_ns()}')
            time.sleep(2 ** attempt)



def compare_catalog(frames, original, root):
    """Preserve official fold labels; disclose upstream catalog discrepancies without relabeling."""
    actual = pd.concat(frames, ignore_index=True)
    if original.Filename.duplicated().any() or set(actual.Filename) != set(original.Filename):
        raise ValueError('Official catalog and split filenames disagree')
    comparison = actual[['Filename', 'Label']].merge(original[['Filename', 'Label']], on='Filename',
                         suffixes=('_split', '_catalog'), validate='one_to_one')
    differences = comparison[comparison.Label_split != comparison.Label_catalog].to_dict('records')
    write_json(Path(root) / 'catalog_label_discrepancies.json', {
        'policy': 'Use unmodified official fold CSV labels; catalog is used for class names only',
        'count': len(differences), 'differences': differences})
    if differences:
        print('UPSTREAM LABEL DISCREPANCIES (fold labels preserved):', differences, flush=True)
    return differences


def prepare(root):
    root = Path(root); data = root / 'data'; labels = data / 'labels'
    archive = data / 'images.zip'
    download('https://zenodo.org/records/7939060/files/images.zip?download=1', archive)
    h = hashlib.md5()
    with archive.open('rb') as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    if h.hexdigest() != MD5:
        raise ValueError('Image archive MD5 mismatch; existing file preserved')
    for name in ('labels', 'train_subset0', 'val_subset0', 'test_subset0'):
        download(f'https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels/{name}.csv', labels / f'{name}.csv')
    target = data / 'extracted'
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            path = target / member.filename
            if not path.resolve().is_relative_to(target.resolve()):
                raise ValueError('Unsafe archive member')
            if not path.exists():
                z.extract(member, target)
    original = pd.read_csv(labels / 'labels.csv')
    sample = next(target.rglob(str(original.Filename.iloc[0])))
    images = sample.parent
    frames = dataset.load_split(labels)
    result = dataset.check_split(*frames, images)
    compare_catalog(frames, original, root)
    write_json(root / 'data_checksums.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in labels.glob('*.csv')})
    write_json(root / 'split_checks.json', result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4))
    pd.DataFrame(result['per_class']).rename(index=dict(enumerate(dataset.CLASS_NAMES))).plot.bar(ax=ax)
    ax.set_ylabel('Number of images'); fig.tight_layout(); fig.savefig(root / 'class_distribution.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(9, 3, figsize=(9, 24))
    # EDA images come from TRAIN only, avoiding inspection of held-out test examples.
    for label in range(9):
        examples = frames[0][frames[0].Label == label].head(3)
        for ax, (_, row) in zip(axes[label], examples.iterrows()):
            with Image.open(images / row.Filename) as image:
                ax.imshow(image.convert('RGB'))
            ax.set_title(dataset.CLASS_NAMES[label]); ax.axis('off')
    fig.tight_layout(); fig.savefig(root / 'sample_images.png', dpi=100); plt.close(fig)
    return str(images), str(labels)


def pipeline_checks(images_dir, labels_dir, out_dir, device='cuda'):
    """Exercise data/labels/gradient flow using a tiny sanity model before full experiments."""
    import matplotlib.pyplot as plt
    set_seed(0)
    df = dataset.load_split(labels_dir)[0].groupby('Label').head(1).reset_index(drop=True)
    loader = dataset.make_loader(df, images_dir, dataset.build_transforms(True, 64), 9, False, num_workers=0)
    x, y, names = next(iter(loader)); x, y = x.to(device), y.to(device)
    tiny = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 64 * 64, 9)).to(device)
    # Small random head, so uniform-confidence CE is approximately ln(9).
    torch.nn.init.normal_(tiny[1].weight, std=1e-4); torch.nn.init.zeros_(tiny[1].bias)
    criterion = torch.nn.CrossEntropyLoss(); optimizer = torch.optim.Adam(tiny.parameters(), lr=1e-3)
    initial = float(criterion(tiny(x), y).detach())
    for step in range(200):
        optimizer.zero_grad(); loss = criterion(tiny(x), y); loss.backward(); optimizer.step()
        if float(loss.detach()) < 0.01:
            break
    final = float(criterion(tiny(x), y).detach())
    checks = {'model': 'tiny linear pipeline sanity model, not a backbone result', 'initial_ce': initial,
              'expected_ce': float(np.log(9)), 'overfit_ce': final, 'steps': step + 1, 'filenames': names}
    if abs(initial - np.log(9)) > 0.3 or final > 0.05:
        raise RuntimeError(f'Pipeline sanity failed: {checks}')
    write_json(Path(out_dir) / 'pipeline_checks.json', checks)
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for i, ax in enumerate(axes.flat):
        image = x[i].cpu().permute(1, 2, 0).numpy() * dataset.IMAGENET_STD + dataset.IMAGENET_MEAN
        ax.imshow(np.clip(image, 0, 1)); ax.set_title(dataset.CLASS_NAMES[int(y[i])]); ax.axis('off')
    fig.tight_layout(); fig.savefig(Path(out_dir) / 'augmented_images.png', dpi=120); plt.close(fig)
    print('Pipeline sanity:', checks, flush=True)
