"""
TrajectoryLNAS Neural Architecture Search (DARTS Engine)
========================================================
Sole Responsibility: Searches for the optimal Lightweight Trajectory Prediction
architecture via Differentiable Bi-Level Optimization (DARTS).
Returns the discovered operator indices (temporal, interaction, map).

Author: Alireza Ghafouri (Student ID: 4031631209)
"""

import os
import sys
import time
import argparse
from typing import Optional, Tuple

# Ensure python_src directory is in sys.path
_curr_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_curr_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from config.search_config import SearchConfig
from core.darts_searcher import DARTSSearcher
from core.logger import DualLogger


def run_architecture_search(
    search_cfg: Optional[SearchConfig] = None,
    dataset: str = "nuscenes",
    train_path: Optional[str] = None,
    val_path: Optional[str] = None,
    epochs: int = 30,
    warmup_epochs: int = 5,
    batch_size: int = 128,
    lambda_val: float = 0.01,
    search_samples_percent: Optional[float] = 20.0,
    log_file: Optional[str] = "checkpoints/search_execution.log"
) -> Tuple[Tuple[int, int, int], DARTSSearcher]:
    """
    Executes bi-level differentiable search and returns (best_indices, searcher_instance).
    Follows Dependency Injection: accepts pre-configured SearchConfig object directly.
    """
    logger = None
    original_stdout = sys.stdout
    if log_file:
        logger = DualLogger(log_file)
        sys.stdout = logger
        print(f"[Logger] Real-time architecture search log: {log_file}\n")

    t_start = time.time()

    try:
        cfg = search_cfg or SearchConfig(
            dataset=dataset,
            train_path=train_path,
            val_path=val_path,
            epochs=epochs,
            warmup_epochs=warmup_epochs,
            batch_size=batch_size,
            lambda_flops=lambda_val,
            search_samples_percent=search_samples_percent
        )

        searcher = DARTSSearcher(search_config=cfg)
        best_indices = searcher.run_search()

        elapsed = time.time() - t_start
        searcher.elapsed_seconds = elapsed
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        print("=" * 80)
        print(" [Step 1/3] Architecture Search Completed Successfully!")
        print(f" Search Runtime: {hours}h {minutes:02d}m {seconds:02d}s ({elapsed:.1f} seconds)")
        print(f" Discovered Operator Indices: {best_indices}")
        if log_file:
            print(f" Full execution trace recorded in: {log_file}")
        print("=" * 80)


        return best_indices, searcher

    finally:
        if logger:
            sys.stdout = original_stdout
            logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrajectoryLNAS Architecture Search (DARTS)")
    parser.add_argument("--dataset", type=str, default="nuscenes", choices=["nuscenes", "argoverse"])
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lambda_val", type=float, default=0.01)
    parser.add_argument("--log_file", type=str, default="checkpoints/search_execution.log")
    args = parser.parse_args()

    run_architecture_search(
        dataset=args.dataset,
        train_path=args.train_path,
        val_path=args.val_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lambda_val=args.lambda_val,
        log_file=args.log_file
    )
