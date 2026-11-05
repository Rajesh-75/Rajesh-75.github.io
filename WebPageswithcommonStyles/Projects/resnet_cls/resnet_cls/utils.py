"""Seeding, logging and safe checkpoint writing."""
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic   # benchmark = faster, non-deterministic


def seed_worker(worker_id: int) -> None:
    """Give every DataLoader worker a distinct, reproducible augmentation seed."""
    s = torch.initial_seed() % 2**32
    random.seed(s)
    np.random.seed(s)


def setup_logging(output_dir) -> logging.Logger:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(Path(output_dir) / "train.log")],
    )
    return logging.getLogger("resnet_cls")


def atomic_save(obj, path) -> None:
    """Write to a temp file then rename, so a crash never leaves a corrupt checkpoint."""
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)
