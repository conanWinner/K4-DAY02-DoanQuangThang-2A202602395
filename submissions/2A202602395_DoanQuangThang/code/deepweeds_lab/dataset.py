"""Official splits, deterministic evaluation and seeded data loading."""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T

NUM_CLASSES = 9
CLASS_NAMES = ['Chinee Apple', 'Lantana', 'Parkinsonia', 'Parthenium', 'Prickly Acacia',
               'Rubber Vine', 'Siam Weed', 'Snake Weed', 'Negatives']
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir, fold=0):
    if fold not in range(5):
        raise ValueError('Fold must be in 0..4')
    return tuple(pd.read_csv(Path(labels_dir) / f'{s}_subset{fold}.csv') for s in ('train', 'val', 'test'))


def check_split(train_df, val_df, test_df, images_dir):
    frames = dict(zip(('train', 'val', 'test'), (train_df, val_df, test_df)))
    sets, result = {}, {'n': {}, 'per_class': {}, 'overlap': {}}
    for name, df in frames.items():
        if not {'Filename', 'Label'}.issubset(df.columns):
            raise ValueError(f'{name}: missing CSV columns')
        if df.Filename.isna().any() or df.Filename.duplicated().any():
            raise ValueError(f'{name}: missing/duplicate filename')
        if df.Label.isna().any() or not df.Label.isin(range(9)).all():
            raise ValueError(f'{name}: invalid label')
        sets[name] = set(df.Filename)
        result['n'][name] = len(df)
        result['per_class'][name] = df.Label.value_counts().reindex(range(9), fill_value=0).to_dict()
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        n = len(sets[a] & sets[b]); result['overlap'][f'{a}_{b}'] = n
        if n:
            raise ValueError(f'Overlapping {a}/{b}: {n}')
    union = set.union(*sets.values())
    if len(union) != 17509:
        raise ValueError(f'Expected 17509 distinct images, got {len(union)}')
    for name, expected in zip(frames, (0.6, 0.2, 0.2)):
        if abs(len(frames[name]) / len(union) - expected) > 0.01:
            raise ValueError(f'{name}: split ratio differs by >1 percentage point')
    missing = [f for f in union if not (Path(images_dir) / f).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} images missing, e.g. {missing[:3]}')
    print('Split checks:', result, flush=True)
    return result


def build_transforms(train, img_size=224, aug='basic', mean=IMAGENET_MEAN, std=IMAGENET_STD):
    interpolation = T.InterpolationMode.BICUBIC
    if train:
        ops = [T.RandomResizedCrop(img_size, interpolation=interpolation), T.RandomHorizontalFlip()]
        if aug == 'color':
            ops += [T.ColorJitter(0.2, 0.2, 0.2, 0.05)]
        elif aug == 'trivial':
            ops += [T.TrivialAugmentWide(interpolation=interpolation)]
        elif aug == 'randaug':
            ops += [T.RandAugment(interpolation=interpolation)]
        elif aug != 'basic':
            raise ValueError(f'Unknown augmentation: {aug}')
    else:
        # Same deterministic resize/crop rule for validation and final test.
        ops = [T.Resize(round(img_size * 256 / 224), interpolation=interpolation), T.CenterCrop(img_size)]
    return T.Compose(ops + [T.ToTensor(), T.Normalize(mean, std)])


class DeepWeedsDataset(Dataset):
    def __init__(self, df, images_dir, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.images_dir, self.transform = Path(images_dir), transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        with Image.open(self.images_dir / row.Filename) as image:
            image = image.convert('RGB')
            image = self.transform(image) if self.transform else T.ToTensor()(image)
        return image, int(row.Label), str(row.Filename)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed); random.seed(seed)


def make_loader(df, images_dir, transform, batch_size, train, sampler=None, num_workers=2, seed=0):
    generator = torch.Generator().manual_seed(seed)
    ds = DeepWeedsDataset(df, images_dir, transform)
    sample = None
    if sampler is not None:
        if sampler != 'balanced' or not train:
            raise ValueError('Balanced sampler is only supported for training')
        counts = df.Label.value_counts()
        sample = WeightedRandomSampler([1 / counts[y] for y in df.Label], len(df), replacement=True,
                                       generator=generator)
    return DataLoader(ds, batch_size=batch_size, shuffle=train and sample is None, sampler=sample,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      drop_last=train and len(ds) % batch_size == 1, generator=generator,
                      worker_init_fn=seed_worker, persistent_workers=num_workers > 0)
