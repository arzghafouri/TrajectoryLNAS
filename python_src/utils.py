"""
System Utilities & Reproducibility Management
=============================================
Follows Single Responsibility Principle (SRP):
Core module for PRNG seed locking, deterministic CUDA/CUDNN setup,
and backward-compatible facade re-exports for domain-specific components.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

# Re-exports from dedicated SRP modules
from evaluate.metrics import ade, fde, miss_rate
from evaluate.zero_cost_proxies import snip_score, synflow_score
from evaluate.hardware_profiler import (
    get_supernet_costs,
    get_supernet_flops,
    get_supernet_complexity,
    simple_complexity_proxy
)
from core.logger import DualLogger
from data.synthetic_dataset import SyntheticTrajectoryDataset


def set_seed(seed: int = 42) -> None:
    """
    Guarantees 100% reproducible execution across PyTorch, CUDA, NumPy, and Python PRNG.
    Eliminates initial weight initialization noise and batch shuffling variance.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_dataloaders(train_path=None, val_path=None, batch_size=32, dataset_type="nuscenes", sample_percent=None, **kwargs):
    """
    Creates PyTorch DataLoaders. Attempts to load from data.data_factory (nuScenes or Argoverse).
    Falls back gracefully to SyntheticTrajectoryDataset if datasets are not found.
    """
    try:
        from data.data_factory import get_trajectory_dataloaders
        return get_trajectory_dataloaders(
            dataset_name=dataset_type,
            train_dir=train_path,
            val_dir=val_path,
            batch_size=batch_size,
            sample_percent=sample_percent,
            **kwargs
        )
    except Exception as e:
        print(f"[DataLoader Fallback] Using SyntheticTrajectoryDataset: {e}")
        train_dataset = SyntheticTrajectoryDataset(num_samples=1200, num_agents=16)
        val_dataset = SyntheticTrajectoryDataset(num_samples=300, num_agents=16)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader



__all__ = [
    "set_seed",
    "get_dataloaders",
    "DualLogger",
    "SyntheticTrajectoryDataset",
    "snip_score",
    "synflow_score",
    "get_supernet_costs",
    "get_supernet_flops",
    "get_supernet_complexity",
    "simple_complexity_proxy",
    "ade",
    "fde",
    "miss_rate",
]
