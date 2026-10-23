"""
Unified Data Factory for TrajectoryLNAS
=======================================
Follows Open/Closed Principle (OCP) and Single Responsibility Principle (SRP):
Employs an extensible DatasetRegistry pattern. New autonomous driving benchmarks
(e.g., Waymo Open, Lyft Level 5) can register without altering core factory logic.
"""

import os
from typing import Callable, Dict, Any, Tuple, Optional
from torch.utils.data import DataLoader

from data.synthetic_dataset import SyntheticTrajectoryDataset

DEFAULT_DATASET_PATHS = {
    "argoverse": {
        "train": "data/argoverse/train/data",
        "val": "data/argoverse/val/data",
        "download_url": "https://www.argoverse.org/av1.html",
        "description": "Argoverse 1 Motion Forecasting (205k train sequences, CSV format, 10Hz)"
    },
    "nuscenes": {
        "train": r"F:\Program\Dataset\nuScenes\v1.0-trainval",
        "val": r"F:\Program\Dataset\nuScenes\v1.0-trainval",
        "download_url": "https://www.nuscenes.org/download",
        "description": "nuScenes Prediction Benchmark (v1.0-trainval metadata + map expansion)"
    }
}


class DatasetRegistry:
    """
    Registry for dataset loader builders.
    Enforces the Open/Closed Principle: Open for new dataset extensions via registration,
    Closed for modification in client retrieval code.
    """
    _registry: Dict[str, Callable[..., Tuple[DataLoader, DataLoader]]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a dataset builder function."""
        def decorator(builder_fn: Callable[..., Tuple[DataLoader, DataLoader]]):
            cls._registry[name.lower()] = builder_fn
            return builder_fn
        return decorator

    @classmethod
    def get_builder(cls, name: Optional[str]) -> Optional[Callable[..., Tuple[DataLoader, DataLoader]]]:
        ds_key = str(name).lower() if name else "nuscenes"
        return cls._registry.get(ds_key)


    @classmethod
    def registered_datasets(cls):
        return list(cls._registry.keys())


# -------------------------------------------------------------------------
# Concrete Dataset Builders Registered via OCP Decorator
# -------------------------------------------------------------------------

@DatasetRegistry.register("argoverse")
def _build_argoverse(
    train_dir: Optional[str] = None,
    val_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 2,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    from data.argoverse_dataset import build_argoverse_dataloader
    t_dir = train_dir or DEFAULT_DATASET_PATHS["argoverse"]["train"]
    v_dir = val_dir or DEFAULT_DATASET_PATHS["argoverse"]["val"]
    
    train_loader = build_argoverse_dataloader(t_dir, split="train", batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = build_argoverse_dataloader(v_dir, split="val", batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


@DatasetRegistry.register("nuscenes")
def _build_nuscenes(
    train_dir: Optional[str] = None,
    val_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 2,
    sample_percent: Optional[float] = None,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    from data.nuscenes_dataset import build_nuscenes_dataloader
    t_dir = train_dir or DEFAULT_DATASET_PATHS["nuscenes"]["train"]
    v_dir = val_dir or DEFAULT_DATASET_PATHS["nuscenes"]["val"]
    
    # Extract sample_percent if passed in kwargs under different names
    if sample_percent is None:
        sample_percent = kwargs.get("search_samples_percent", kwargs.get("train_samples_percent", kwargs.get("sample_percent", kwargs.get("max_samples", None))))

    train_loader = build_nuscenes_dataloader(
        t_dir, split="train", batch_size=batch_size, shuffle=True, num_workers=num_workers, sample_percent=sample_percent
    )
    val_loader = build_nuscenes_dataloader(
        v_dir, split="val", batch_size=batch_size, shuffle=False, num_workers=num_workers, sample_percent=sample_percent
    )
    return train_loader, val_loader


@DatasetRegistry.register("synthetic")
def _build_synthetic(
    batch_size: int = 32,
    train_samples: int = 1200,
    val_samples: int = 300,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    sample_pct = kwargs.get("sample_percent", kwargs.get("search_samples_percent", kwargs.get("train_samples_percent", None)))
    if sample_pct is not None and sample_pct > 0:
        train_samples = max(100, int(train_samples * (sample_pct / 100.0)))
        val_samples = max(50, int(val_samples * (sample_pct / 100.0)))

    train_ds = SyntheticTrajectoryDataset(num_samples=train_samples, num_agents=16)
    val_ds = SyntheticTrajectoryDataset(num_samples=val_samples, num_agents=16)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# -------------------------------------------------------------------------
# Public Factory API
# -------------------------------------------------------------------------

def get_trajectory_dataloaders(
    dataset_name: str = "nuscenes",
    train_dir: Optional[str] = None,
    val_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 2,
    sample_percent: Optional[float] = None,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Unified DataLoader provider.
    Dispatches to the registered dataset builder without rigid conditional logic.
    """
    raw_ds = kwargs.get("dataset_type") or kwargs.get("dataset_name") or dataset_name or "nuscenes"
    ds_name = str(raw_ds).lower() if raw_ds else "nuscenes"
    t_path = kwargs.get("train_path", train_dir)
    v_path = kwargs.get("val_path", val_dir)


    builder = DatasetRegistry.get_builder(ds_name)
    if builder is None:
        raise ValueError(
            f"Unsupported dataset: '{ds_name}'. "
            f"Available registered datasets: {DatasetRegistry.registered_datasets()}"
        )

    return builder(
        train_dir=t_path,
        val_dir=v_path,
        batch_size=batch_size,
        num_workers=num_workers,
        sample_percent=sample_percent,
        **kwargs
    )

