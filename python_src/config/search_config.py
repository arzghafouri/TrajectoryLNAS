r"""
Configuration Dataclasses for TrajectoryLNAS
============================================
Defines typed, immutable configuration classes for Differentiable Neural Architecture Search
and final standalone model training.
Optimized defaults for NVIDIA GeForce RTX 4060 Ti (8GB VRAM), 32GB System RAM, and Intel Core i7 13th Gen:
  - Target Effective Batch Size: 128
  - DARTS Search Phase: batch_size = 32 (Physical) | grad_accum_steps = 4 (Effective = 128)
  - Final Retrain Phase: batch_size = 64 (Physical) | grad_accum_steps = 2 (Effective = 128)
  - Full interaction capacity: max_agents = 16, max_lanes = 20
  - System memory capacity: 32GB RAM enables larger dataset caches and higher worker throughput
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union

@dataclass
class ModelConfig:
    """Dimensions and hyperparameters of the SuperNet and Final Model."""
    d_model: int = 128        # Unchanged to maintain high expressive capacity
    past_len: int = 20        # 2.0s observation at 10Hz
    future_len: int = 30      # 3.0s prediction at 10Hz
    num_modes: int = 6        # Multi-modal trajectory hypotheses (K=6 for SOTA / nuScenes)
    max_agents: int = 16      # Full interaction capacity (1 target + 15 neighbors)
    max_lanes: int = 20       # Full HD map capacity (20 vectorized lane polylines)
    lane_pts: int = 10        # Points per polyline
    dropout: float = 0.1

@dataclass
class DataConfig:
    r"""
    Configuration for dataset extraction, preprocessing, and synthetic fallback.
    Single Source of Truth for data preparation parameters.
    
    Attributes:
        dataroot: Path to raw nuScenes dataset directory.
        output_dir: Destination path for train_sequences.json and val_sequences.json.
                    Default: r"F:\Program\Dataset\nuScenes\v1.0-trainval"
        target_count: Target number of trajectory sequences to extract (default: 12800).
                      If set to None, ALL available samples across all scenes are extracted without truncation.
        category: Agent category or list of categories to filter.
        mode: Extraction policy ('auto', 'real', 'synthetic').
        seed: Deterministic random seed for sequence shuffling.
        max_train_scenes: Number of scenes to extract for training (None = all scenes).
        max_val_scenes: Number of scenes to extract for validation (None = all scenes).
    """
    dataroot: str = r"F:\Program\Dataset\nuScenes"
    output_dir: str = r"F:\Program\Dataset\nuScenes\v1.0-trainval"
    target_count: Optional[int] = 12800  # Elevated capacity for 32GB System RAM (or None for ALL)
    category: Optional[Union[str, List[str]]] = None  # None = all categories
    mode: str = "auto"
    seed: int = 42
    max_train_scenes: Optional[int] = None
    max_val_scenes: Optional[int] = None

@dataclass
class SearchConfig:
    """Hyperparameters for DARTS Bi-Level Optimization (Physical=32, Accum=4 -> Effective=128)."""
    dataset: str = "nuscenes"
    train_path: Optional[str] = None
    val_path: Optional[str] = None
    seed: int = 42            # Reproducibility seed for deterministic weight initialization and shuffling
    warmup_epochs: int = 5    # Warmup epochs where alpha is frozen
    epochs: int = 35          # 35 epochs for sharper architectural operator separation
    batch_size: int = 32      # Physical batch size 32 for DARTS search phase
    grad_accum_steps: int = 4 # Effective batch size = 32 * 4 = 128
    val_batches_for_alpha: int = 4 # 4 * 32 = 128 validation sequences to match effective batch size
    use_amp: bool = True      # Automatic Mixed Precision (FP16) on Ada Lovelace Tensor Cores
    lr_weights: float = 7.5e-4 # Scaled for effective batch size 128
    lr_alphas: float = 5e-3
    weight_decay: float = 1e-4
    alpha_weight_decay: float = 1e-3
    lambda_flops: float = 0.01
    temperature_start: float = 1.0
    temperature_min: float = 0.2
    search_samples_percent: Optional[float] = 20.0  # Dynamic percentage of total dataset used for search (e.g. 20.0 = 20%)
    device: str = "cuda"

@dataclass
class TrainConfig:
    """Hyperparameters for Standalone Final Model Retraining (Physical=64, Accum=2 -> Effective=128)."""
    dataset: str = "nuscenes"
    train_path: Optional[str] = None
    val_path: Optional[str] = None
    seed: int = 42            # Reproducibility seed for retraining
    epochs: int = 50          # Higher ceiling for final model
    batch_size: int = 64      # Physical batch size 64 for final retraining phase
    grad_accum_steps: int = 2 # Effective batch size = 64 * 2 = 128
    use_amp: bool = True      # FP16 acceleration on Ada Lovelace Tensor Cores
    lr: float = 7.5e-4        # Scaled learning rate for effective batch size 128
    weight_decay: float = 1e-4
    clip_grad_norm: float = 5.0
    early_stopping_patience: int = 8      # Number of epochs to wait for improvement
    early_stopping_min_delta: float = 0.002 # Minimum change in validation ADE (2mm)
    train_samples_percent: Optional[float] = None # Full dataset (100%) by default for final training
    save_path: str = "checkpoints/best_trajectory_lnas.pt"
    device: str = "cuda"


