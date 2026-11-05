"""Single source of truth for every hyperparameter; saved next to each run."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass


@dataclass
class Config:
    # data
    data_dir: str = "data"            # expects train/ val/ and optionally test/
    output_dir: str = "runs/exp1"
    grayscale: bool = True            # True for X-ray; False for fundus/derm/histology
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    # model
    arch: str = "resnet50"            # resnet50 | resnet101
    pretrained: bool = True
    dropout: float = 0.3
    # optimisation
    head_epochs: int = 3
    full_epochs: int = 12
    lr_head: float = 1e-3
    lr_layer3: float = 1e-5
    lr_layer4: float = 5e-5
    lr_fc: float = 5e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 4                 # early stopping (epochs without val-AUC gain)
    # system
    seed: int = 42
    deterministic: bool = False       # True = bit-reproducible but slower
    amp: bool = True
    resume: str = ""                  # path to last.pt to resume from


def parse_args(argv=None) -> Config:
    """Auto-generate a CLI flag for every Config field (--data-dir, --amp/--no-amp, ...)."""
    parser = argparse.ArgumentParser(description="Fine-tune ResNet for medical image classification")
    for name, default in asdict(Config()).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        else:
            parser.add_argument(flag, type=type(default), default=default)
    return Config(**vars(parser.parse_args(argv)))
