"""
TrajectoryLNAS Standalone Final Retraining
==========================================
Sole Responsibility: Trains the discretized FinalTrajectoryModel discovered by DARTS.

Author: Alireza Ghafouri (Student ID: 4031631209)
"""

import os
import sys
import argparse
from typing import Optional, Tuple

# Enable running both from root and from train/ directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.search_config import TrainConfig
from core.final_trainer import FinalModelTrainer
from models.final_model import FinalTrajectoryModel


def run_final_training(
    best_op_indices: Tuple[int, int, int] = (3, 4, 1),
    train_cfg: Optional[TrainConfig] = None,
    dataset: str = "nuscenes",
    train_path: Optional[str] = None,
    val_path: Optional[str] = None,
    epochs: int = 50,
    batch_size: int = 128,
    grad_accum_steps: int = 1,
    lr: float = 5e-4,
    patience: int = 8,
    min_delta: float = 0.002,
    train_samples_percent: Optional[float] = None,
    save_path: str = "checkpoints/best_trajectory_lnas.pt"
) -> Tuple[FinalTrajectoryModel, FinalModelTrainer]:
    """
    Executes standalone retraining of the winning architecture with Early Stopping.
    Follows Dependency Injection: accepts pre-configured TrainConfig object directly.
    """
    cfg = train_cfg or TrainConfig(
        dataset=dataset,
        train_path=train_path,
        val_path=val_path,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        lr=lr,
        early_stopping_patience=patience,
        early_stopping_min_delta=min_delta,
        train_samples_percent=train_samples_percent,
        save_path=save_path
    )

    trainer = FinalModelTrainer(
        best_op_indices=best_op_indices,
        train_config=cfg
    )
    final_model = trainer.train_and_evaluate()
    return final_model, trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrajectoryLNAS Standalone Final Training")
    parser.add_argument("--dataset", type=str, default="nuscenes", choices=["nuscenes", "argoverse"])
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--temp_op", type=int, default=3)
    parser.add_argument("--inter_op", type=int, default=4)
    parser.add_argument("--map_op", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_path", type=str, default="checkpoints/best_trajectory_lnas.pt")
    args = parser.parse_args()

    run_final_training(
        best_op_indices=(args.temp_op, args.inter_op, args.map_op),
        dataset=args.dataset,
        train_path=args.train_path,
        val_path=args.val_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path
    )
