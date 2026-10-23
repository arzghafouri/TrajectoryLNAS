"""
TrajectoryLNAS Main Orchestrator Pipeline
=========================================
Master Entry Point executing the complete 3-Stage Scientific Workflow:
  Stage 0: Automatic Dataset Verification & Preparation (data/prepare_data.py)
  Stage 1: Differentiable Neural Architecture Search (DARTS in train/search.py)
  Stage 2: Standalone Retraining of Discovered Model (train/train_final.py)
  Stage 3: Full Academic Evaluation, SOTA Benchmarking & Metric Plotting (evaluate/)

All execution logs, metrics summaries, and visualization plots are saved into report/.

Author: Alireza Ghafouri (Student ID: 4031631209)
Advisors: Dr. Zeinab Ghasemi Naraaghi & Dr. Zeinab Abedini
"""

import os
import sys
import time
import argparse
import json

from typing import Optional
from config.search_config import SearchConfig, TrainConfig, DataConfig
from train.search import run_architecture_search
from train.train_final import run_final_training
from evaluate import TrajectoryEvaluator
from core.logger import DualLogger
from data.data_factory import DEFAULT_DATASET_PATHS
from data.prepare_data import DataPreparationPipeline, run_official_nuscenes_extraction, run_synthetic_preparation
from evaluate.plotter import plot_horizon_error_curve, plot_benchmark_comparison


def ensure_dataset_ready(dataset: str, train_path: str = None, count: Optional[int] = 6400):
    """
    Checks if train and val dataset files exist.
    If not found, invokes DataPreparationPipeline which checks maps/expansion availability
    and extracts shuffled sequences (if count is None, extracts ALL available sequences).
    """
    if dataset == "nuscenes":
        t_dir = train_path or DEFAULT_DATASET_PATHS["nuscenes"]["train"]
        train_json = os.path.join(t_dir, "train_sequences.json")
        val_json = os.path.join(t_dir, "val_sequences.json")

        if not (os.path.exists(train_json) and os.path.exists(val_json)):
            count_desc = f"{count:,} shuffled sequences" if count is not None else "ALL available sequences (No truncation)"
            print(f"[Stage 0: Data Verification] Dataset sequences missing in: {t_dir}")
            print(f"  -> Executing DataPreparationPipeline to prepare {count_desc}...")
            raw_root = os.path.dirname(t_dir) if "v1.0-trainval" in t_dir else t_dir
            pipeline = DataPreparationPipeline()
            pipeline.prepare(dataroot=raw_root, output_dir=t_dir, target_count=count, mode="auto", seed=42)
        else:
            print(f"[Stage 0: Data Verification] ✓ Verified dataset files in: {t_dir}")


def main_pipeline(
    search_cfg: SearchConfig = None,
    train_cfg: TrainConfig = None,
    data_cfg: DataConfig = None,
    dataset: str = None,
    train_path: str = None,
    val_path: str = None,
    search_epochs: int = None,
    warmup_epochs: int = None,
    final_epochs: int = None,
    patience: int = None,
    min_delta: float = None,
    batch_size: int = None,
    search_batch_size: int = None,
    train_batch_size: int = None,
    grad_accum_steps: int = None,
    search_grad_accum: int = None,
    train_grad_accum: int = None,
    lambda_val: float = None,
    train_samples_percent: float = None,
    search_samples_percent: float = None,
    report_dir: str = "report",
    output_checkpoint: str = None,
    skip_search: bool = False,
    temp_op: int = 3,
    inter_op: int = 4,
    map_op: int = 1
):
    """
    Executes the master pipeline, routing all logs and outputs to report/
    and model weights to checkpoints/

    Follows clean Dependency Injection: SearchConfig, TrainConfig, and DataConfig in config/search_config.py
    serve as the single source of truth for all hyperparameters. Any parameter changed in
    search_config.py is automatically respected unless explicitly overridden via CLI.
    """
    # 1. Instantiate configuration objects from search_config.py (Source of Truth)
    if search_cfg is None:
        search_cfg = SearchConfig()
    if train_cfg is None:
        train_cfg = TrainConfig()
    if data_cfg is None:
        data_cfg = DataConfig()

    # 2. Apply explicit overrides only if provided (not None)
    if dataset is not None:
        search_cfg.dataset = dataset
        train_cfg.dataset = dataset
    if train_path is not None:
        search_cfg.train_path = train_path
        train_cfg.train_path = train_path
    if val_path is not None:
        search_cfg.val_path = val_path
        train_cfg.val_path = val_path
    if search_epochs is not None:
        search_cfg.epochs = search_epochs
    if warmup_epochs is not None:
        search_cfg.warmup_epochs = warmup_epochs
    if final_epochs is not None:
        train_cfg.epochs = final_epochs
    if patience is not None:
        train_cfg.early_stopping_patience = patience
    if min_delta is not None:
        train_cfg.early_stopping_min_delta = min_delta

    # Distinct Search vs Final Retrain Batch Sizes & Accumulation
    if search_batch_size is not None:
        search_cfg.batch_size = search_batch_size
    elif batch_size is not None:
        search_cfg.batch_size = batch_size

    if train_batch_size is not None:
        train_cfg.batch_size = train_batch_size
    elif batch_size is not None:
        train_cfg.batch_size = batch_size

    if search_grad_accum is not None:
        search_cfg.grad_accum_steps = search_grad_accum
    elif grad_accum_steps is not None:
        search_cfg.grad_accum_steps = grad_accum_steps

    if train_grad_accum is not None:
        train_cfg.grad_accum_steps = train_grad_accum
    elif grad_accum_steps is not None:
        train_cfg.grad_accum_steps = grad_accum_steps

    if lambda_val is not None:
        search_cfg.lambda_flops = lambda_val
    if train_samples_percent is not None:
        train_cfg.train_samples_percent = train_samples_percent
    if search_samples_percent is not None:
        search_cfg.search_samples_percent = search_samples_percent
    if output_checkpoint is not None:
        train_cfg.save_path = output_checkpoint

    active_dataset = search_cfg.dataset
    search_effective_batch = search_cfg.batch_size * search_cfg.grad_accum_steps
    train_effective_batch = train_cfg.batch_size * train_cfg.grad_accum_steps
    active_checkpoint = train_cfg.save_path

    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(os.path.join(report_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.dirname(active_checkpoint) if os.path.dirname(active_checkpoint) else "checkpoints", exist_ok=True)

    log_file = os.path.join(report_dir, "search_execution.log")
    export_json = os.path.join(report_dir, f"evaluation_results_{active_dataset}.json")

    from utils import set_seed
    set_seed(getattr(search_cfg, "seed", 42))

    logger = DualLogger(log_file)
    orig_stdout = sys.stdout
    sys.stdout = logger

    t_global_start = time.time()

    data_count_str = f"{data_cfg.target_count:,}" if data_cfg.target_count is not None else "ALL available sequences"

    train_pct_str = f"{train_cfg.train_samples_percent}%" if getattr(train_cfg, "train_samples_percent", None) is not None else "100% (Full Dataset)"
    search_pct_str = f"{search_cfg.search_samples_percent}%" if getattr(search_cfg, "search_samples_percent", None) is not None else "100% (Full Dataset)"
    print("=" * 80)
    print(" TRAJECTORY LNAS: MASTER ORCHESTRATION PIPELINE")
    print(f" Target Dataset:      {active_dataset.upper()}")
    print(f" DARTS Search Epochs: {search_cfg.epochs} (Warmup: {getattr(search_cfg, 'warmup_epochs', 5)} epochs) | Final Retrain Ceiling: {train_cfg.epochs}")
    print(f" Reproducibility:     Seed = {getattr(search_cfg, 'seed', 42)} (100% Deterministic Execution)")
    print(f" Dataset Subsampling: {train_pct_str} Sequences (Uniformly Shuffled)")
    print(f" DARTS Subsampling:   {search_pct_str} Sequences (Search Phase)")
    print(f" Early Stopping:      Patience = {train_cfg.early_stopping_patience} epochs | Min Delta = {train_cfg.early_stopping_min_delta}m")
    print(f" Search Batch Size:   {search_cfg.batch_size} (Physical) | Accum Steps: {search_cfg.grad_accum_steps} (Effective: {search_effective_batch})")
    print(f" Train Batch Size:    {train_cfg.batch_size} (Physical) | Accum Steps: {train_cfg.grad_accum_steps} (Effective: {train_effective_batch})")
    print(f" Regularization:      λ = {search_cfg.lambda_flops}")
    print(f" Execution Log File:  {log_file}")
    print("=" * 80)

    try:
        # =========================================================================
        # STAGE 0: Automatic Dataset Verification & Preparation
        # =========================================================================
        ensure_dataset_ready(dataset=active_dataset, train_path=search_cfg.train_path, count=data_cfg.target_count)

        # =========================================================================
        # STAGE 1: Differentiable Neural Architecture Search (DARTS)
        # =========================================================================
        if not skip_search:
            print("[STAGE 1/3] Launching Differentiable Neural Architecture Search...")
            best_ops, searcher_obj = run_architecture_search(
                search_cfg=search_cfg,
                log_file=None
            )
        else:
            best_ops = (temp_op, inter_op, map_op)
            searcher_obj = None
            print(f"[STAGE 1/3 Skipped] Using designated operator indices: {best_ops}")

        # =========================================================================
        # STAGE 2: Standalone Retraining of Discovered Model (with Early Stopping)
        # =========================================================================
        print("[STAGE 2/3] Launching Standalone Training of Discovered Architecture (with Early Stopping)...")
        final_model, trainer_obj = run_final_training(
            best_op_indices=best_ops,
            train_cfg=train_cfg
        )

        # =========================================================================
        # STAGE 3: Academic Evaluation, SOTA Benchmarking & Plotting
        # =========================================================================
        print("[STAGE 3/3] Launching Academic Evaluation & Hardware Benchmarking...")

        evaluator = TrajectoryEvaluator(
            checkpoint_path=active_checkpoint,
            dataset_name=active_dataset,
            best_ops=best_ops
        )
        eval_metrics = evaluator.evaluate_dataset(batch_size=active_batch_size)
        hw_metrics = evaluator.profile_hardware(num_runs=100)
        evaluator.print_evaluation_report(eval_metrics, hw_metrics)
        search_dur = getattr(searcher_obj, "elapsed_seconds", None) if searcher_obj else None
        evaluator.print_sota_comparison(eval_metrics=eval_metrics, hw_metrics=hw_metrics, search_runtime_seconds=search_dur)

        # Save publication plots into report/plots/
        plots_dir = os.path.join(report_dir, "plots")
        plot_horizon_error_curve(eval_results=eval_metrics.get("step_errors", []), dataset=active_dataset, output_dir=plots_dir)
        our_ade = eval_metrics.get("min_ade_3s", eval_metrics.get("ade_3s", 0.0))
        our_fde = eval_metrics.get("min_fde_3s", eval_metrics.get("fde_3s", 0.0))
        plot_benchmark_comparison(dataset=active_dataset, output_dir=plots_dir, our_ade=our_ade, our_fde=our_fde)


        # Export unified real_experiment_results.json for web app binding
        temporal_names = ["Identity", "GRUEncoder", "LSTMEncoder", "TemporalConvEncoder", "TemporalSelfAttention"]
        inter_names = ["Identity", "GCNLayer", "GATLayer", "SocialPooling", "PairwiseAttention"]
        map_names = ["Identity", "LaneAttention", "LaneGraphEncoder", "MapFeatureFusion"]

        import torch
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

        real_experiment_data = {
            "version": "1.0.0",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "thesis": {
                "project": "TrajectoryLNAS",
                "student": "Alireza Ghafouri (4031631209)",
                "supervisors": ["Dr. Zeinab Ghasemi Naraaghi", "Dr. Zeinab Abedini"]
            },
            "environment": {
                "device": device_name,
                "cuda_available": torch.cuda.is_available(),
                "dataset": active_dataset.upper()
            },
            "zero_cost_proxies": {
                "snip": round(searcher_obj.proxies["snip"], 4) if (searcher_obj and hasattr(searcher_obj, "proxies") and searcher_obj.proxies.get("snip") is not None) else None,
                "synflow": round(searcher_obj.proxies["synflow"], 4) if (searcher_obj and hasattr(searcher_obj, "proxies") and searcher_obj.proxies.get("synflow") is not None) else None
            },
            "darts_search": {
                "epochs": search_cfg.epochs if not skip_search else 0,
                "elapsed_seconds": round(search_dur, 2) if search_dur is not None else None,
                "history": searcher_obj.search_history if searcher_obj and hasattr(searcher_obj, "search_history") else []
            },
            "discovered_architecture": {
                "temporal": {"index": best_ops[0], "name": temporal_names[best_ops[0]]},
                "interaction": {"index": best_ops[1], "name": inter_names[best_ops[1]]},
                "map": {"index": best_ops[2], "name": map_names[best_ops[2]]}
            },
            "final_training": {
                "history": trainer_obj.training_history if trainer_obj and hasattr(trainer_obj, "training_history") else [],
                "best_val_ade": round(trainer_obj.best_val_ade, 4) if trainer_obj else eval_metrics.get("min_ade_3s", None),
                "best_val_fde": round(trainer_obj.best_val_fde, 4) if trainer_obj else eval_metrics.get("min_fde_3s", None)
            },
            "edge_hardware_profiling": hw_metrics,
            "evaluation_metrics": eval_metrics
        }

        real_report_path = os.path.join(report_dir, "real_experiment_results.json")
        with open(real_report_path, "w", encoding="utf-8") as f:
            json.dump(real_experiment_data, f, indent=2)
        print(f"[Results Export] Real experiment data successfully saved to: {real_report_path}")

        # Also write backward-compatible evaluation_results_{dataset}.json
        with open(export_json, "w", encoding="utf-8") as f:
            json.dump(real_experiment_data, f, indent=2)

        t_total = time.time() - t_global_start
        h = int(t_total // 3600)
        m = int((t_total % 3600) // 60)
        s = int(t_total % 60)

        print("=" * 80)
        print(" Execution Finished Successfully!")
        print(f" Total Runtime: {h}h {m:02d}m {s:02d}s ({t_total:.1f} seconds)")
        print(f" Full execution trace permanently saved to: {log_file}")
        print(f" Reports & Plots stored in: {report_dir}/")
        print("=" * 80)


    finally:
        sys.stdout = orig_stdout
        logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrajectoryLNAS Master 3-Stage Pipeline")
    parser.add_argument("--dataset", type=str, default=None, choices=["nuscenes", "argoverse"], help="Target dataset (defaults to config/search_config.py)")
    parser.add_argument("--train_path", type=str, default=None, help="Train path override")
    parser.add_argument("--val_path", type=str, default=None, help="Val path override")
    parser.add_argument("--search_epochs", type=int, default=None, help="DARTS search epochs (defaults to SearchConfig.epochs)")
    parser.add_argument("--warmup_epochs", type=int, default=None, help="DARTS warmup epochs where alpha is frozen (defaults to SearchConfig.warmup_epochs)")
    parser.add_argument("--final_epochs", type=int, default=None, help="Final training epochs (defaults to TrainConfig.epochs)")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience (defaults to TrainConfig.early_stopping_patience)")
    parser.add_argument("--min_delta", type=float, default=None, help="Minimum ADE improvement threshold (defaults to TrainConfig.early_stopping_min_delta)")
    parser.add_argument("--batch_size", type=int, default=None, help="Global batch size override (applies to both phases if specific not set)")
    parser.add_argument("--search_batch_size", type=int, default=None, help="Physical batch size for DARTS search phase (defaults to SearchConfig.batch_size=32)")
    parser.add_argument("--train_batch_size", type=int, default=None, help="Physical batch size for final retraining phase (defaults to TrainConfig.batch_size=64)")
    parser.add_argument("--grad_accum", type=int, default=None, help="Global grad accumulation override (applies to both phases if specific not set)")
    parser.add_argument("--search_grad_accum", type=int, default=None, help="Grad accumulation steps for DARTS search (defaults to SearchConfig.grad_accum_steps=4)")
    parser.add_argument("--train_grad_accum", type=int, default=None, help="Grad accumulation steps for final retraining (defaults to TrainConfig.grad_accum_steps=2)")
    parser.add_argument("--lambda_val", type=float, default=None, help="FLOPs penalty weight (defaults to search_config.py)")
    parser.add_argument("--train_samples_percent", type=float, default=None, help="Percentage of dataset for final retraining (defaults to TrainConfig.train_samples_percent)")
    parser.add_argument("--search_samples_percent", type=float, default=None, help="Percentage of dataset for DARTS search (defaults to SearchConfig.search_samples_percent)")
    parser.add_argument("--report_dir", type=str, default="report", help="Destination folder for logs, json, and plots")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint save path (defaults to TrainConfig.save_path)")
    parser.add_argument("--skip_search", action="store_true", help="Skip search and train default operators")
    parser.add_argument("--temp_op", type=int, default=3)
    parser.add_argument("--inter_op", type=int, default=4)
    parser.add_argument("--map_op", type=int, default=1)
    args = parser.parse_args()

    main_pipeline(
        dataset=args.dataset,
        train_path=args.train_path,
        val_path=args.val_path,
        search_epochs=args.search_epochs,
        warmup_epochs=args.warmup_epochs,
        final_epochs=args.final_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        batch_size=args.batch_size,
        search_batch_size=args.search_batch_size,
        train_batch_size=args.train_batch_size,
        grad_accum_steps=args.grad_accum,
        search_grad_accum=args.search_grad_accum,
        train_grad_accum=args.train_grad_accum,
        lambda_val=args.lambda_val,
        train_samples_percent=args.train_samples_percent,
        search_samples_percent=args.search_samples_percent,
        report_dir=args.report_dir,
        output_checkpoint=args.checkpoint,
        skip_search=args.skip_search,
        temp_op=args.temp_op,
        inter_op=args.inter_op,
        map_op=args.map_op
    )
