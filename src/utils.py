from __future__ import annotations
import random
from pathlib import Path
from typing import Any
import numpy as np
import torch
import yaml


def load_config(path: str | Path)->dict[str, Any]:
    path=Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        config=yaml.safe_load(f)
    return config


def set_seed(seed: int)->None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False


def get_device(prefer_cuda: bool=True)->torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
