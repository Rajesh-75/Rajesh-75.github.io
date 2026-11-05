"""Datasets, transforms and loaders. The eval transform is shared with predict.py
so training-time and serving-time preprocessing can never drift apart."""
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .dicom_io import load_image, parse_window
from .manifest import read_manifest
from .utils import seed_worker

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)   # ImageNet statistics


def build_eval_transform(img_size: int, grayscale: bool):
    pre = [transforms.Grayscale(num_output_channels=3)] if grayscale else []
    return transforms.Compose(pre + [
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def build_train_transform(img_size: int, grayscale: bool):
    pre = [transforms.Grayscale(num_output_channels=3)] if grayscale else []
    return transforms.Compose(pre + [
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomRotation(10),
        # No horizontal flip: it would create anatomically impossible chest X-rays.
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class ManifestDataset(Dataset):
    """Same interface as ImageFolder (.classes, .targets), but reads DICOM or ordinary images."""

    def __init__(self, csv_path, split, transform, classes=None, window=None):
        all_classes, rows = read_manifest(csv_path, split)
        self.classes = classes or all_classes
        self.rows, self.transform, self.window = rows, transform, window
        idx = {c: i for i, c in enumerate(self.classes)}
        self.targets = [idx[r["label"]] for r in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        img = load_image(self.rows[i]["path"], self.window)
        return self.transform(img), self.targets[i]


def build_datasets(cfg):
    if cfg.manifest:                                    # manifest / DICOM path
        win = parse_window(cfg.dicom_window)
        t_tf = build_train_transform(cfg.img_size, cfg.grayscale)
        e_tf = build_eval_transform(cfg.img_size, cfg.grayscale)
        train_ds = ManifestDataset(cfg.manifest, "train", t_tf, window=win)
        val_ds = ManifestDataset(cfg.manifest, "val", e_tf, train_ds.classes, win)
        test_ds = ManifestDataset(cfg.manifest, "test", e_tf, train_ds.classes, win)
        if len(train_ds) == 0 or len(val_ds) == 0:
            raise ValueError("Manifest has an empty train or val split")
        return train_ds, val_ds, (test_ds if len(test_ds) else None)

    root = Path(cfg.data_dir)
    train_ds = datasets.ImageFolder(str(root / "train"), build_train_transform(cfg.img_size, cfg.grayscale))
    eval_tf = build_eval_transform(cfg.img_size, cfg.grayscale)
    val_ds = datasets.ImageFolder(str(root / "val"), eval_tf)
    test_ds = datasets.ImageFolder(str(root / "test"), eval_tf) if (root / "test").is_dir() else None

    # Class-index mapping must be identical in every split, or metrics are silently wrong.
    for name, ds in (("val", val_ds), ("test", test_ds)):
        if ds is not None and ds.classes != train_ds.classes:
            raise ValueError(f"Class folders in {name}/ differ from train/: {ds.classes} vs {train_ds.classes}")
    return train_ds, val_ds, test_ds


def make_loader(ds, cfg, train: bool) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    return DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=train, num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker, generator=g,
        drop_last=train,                     # a final batch of size 1 would crash BatchNorm
        persistent_workers=cfg.num_workers > 0,
    )


def class_weights(ds) -> torch.Tensor:
    """Inverse-frequency weights to counter class imbalance."""
    counts = torch.bincount(torch.tensor(ds.targets), minlength=len(ds.classes)).float()
    return len(ds) / (len(ds.classes) * counts.clamp(min=1))
