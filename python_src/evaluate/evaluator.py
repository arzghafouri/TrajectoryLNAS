"""
Evaluator & Performance Profiler
================================
Evaluates FinalTrajectoryModel accuracy and efficiency on real motion forecasting benchmarks.
Supports Multi-Modal Trajectory Evaluation (minADE@6, minFDE@6, MissRate@6) and strict checkpoint auditing.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, Optional, Tuple, List

# Allow running independently: python evaluate/evaluator.py or from root
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import numpy as np

from models.final_model import FinalTrajectoryModel
from config.search_config import ModelConfig
from data.data_factory import get_trajectory_dataloaders as get_dataloaders
from evaluate.metrics import min_ade_k, min_fde_k, miss_rate_k
from evaluate.hardware_profiler import HardwareProfiler
from evaluate.plotter import plot_horizon_error_curve, plot_benchmark_comparison


class TrajectoryEvaluator:
    """
    Production-grade Evaluator for Trajectory Forecasting Architectures.
    Enforces strict checkpoint integrity and genuine multi-modal metric reporting.
    """
    def __init__(
        self,
        checkpoint_path: Optional[str] = "checkpoints/best_trajectory_lnas.pt",
        dataset_name: str = "nuscenes",
        device: Optional[str] = None,
        model_config: Optional[ModelConfig] = None,
        best_ops: Tuple[int, int, int] = (3, 4, 1),
        allow_untrained: bool = False
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")

        self.checkpoint_path = checkpoint_path
        self.dataset_name = dataset_name or "nuscenes"
        self.model_cfg = model_config or ModelConfig()
        self.allow_untrained = allow_untrained

        self.best_ops = best_ops
        self.model = None
        self.checkpoint_metadata = {}

        self._initialize_model()
        self.profiler = HardwareProfiler(
            self.model,
            self.device,
            past_len=self.model_cfg.past_len,
            max_agents=getattr(self.model_cfg, "max_agents", 16),
            max_lanes=getattr(self.model_cfg, "max_lanes", 20),
            lane_pts=getattr(self.model_cfg, "lane_pts", 10)
        )

    def _initialize_model(self):
        """Loads weights from checkpoint or strictly halts if checkpoint is invalid."""
        loaded_checkpoint = False

        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            try:
                try:
                    ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
                except TypeError:
                    ckpt = torch.load(self.checkpoint_path, map_location=self.device)

                self.checkpoint_metadata = ckpt if isinstance(ckpt, dict) else {}
                cfg_info = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
                state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

                temporal_names = ["identity", "gru", "lstm", "temporalconv", "temporalselfattention"]
                interaction_names = ["identity", "gcn", "gat", "socialpooling", "pairwiseattention"]
                map_names = ["identity", "laneattention", "lanegraphencoder", "mapfeaturefusion"]

                best_t = cfg_info.get("best_temporal_idx")
                best_i = cfg_info.get("best_interaction_idx")
                best_m = cfg_info.get("best_map_idx")

                best_t = int(best_t) if best_t is not None else self.best_ops[0]
                best_i = int(best_i) if best_i is not None else self.best_ops[1]
                best_m = int(best_m) if best_m is not None else self.best_ops[2]
                self.best_ops = (best_t, best_i, best_m)

                self.model = FinalTrajectoryModel(
                    best_temporal_idx=best_t,
                    best_interaction_idx=best_i,
                    best_map_idx=best_m,
                    d_model=self.model_cfg.d_model,
                    past_len=self.model_cfg.past_len,
                    future_len=self.model_cfg.future_len,
                    num_modes=getattr(self.model_cfg, "num_modes", 6),
                    max_agents=getattr(self.model_cfg, "max_agents", 16),
                    max_lanes=getattr(self.model_cfg, "max_lanes", 20),
                    lane_pts=getattr(self.model_cfg, "lane_pts", 10)
                ).to(self.device)

                self.model.load_state_dict(state_dict)
                loaded_checkpoint = True
                print(f"[Evaluator] Successfully loaded checkpoint: {self.checkpoint_path}")
                print(f"[Evaluator] Architecture: Temporal={temporal_names[best_t].upper()} (idx {best_t}) | "
                      f"Interaction={interaction_names[best_i].upper()} (idx {best_i}) | "
                      f"Map={map_names[best_m].upper()} (idx {best_m})")
            except Exception as e:
                if not self.allow_untrained:
                    raise RuntimeError(f"[Evaluator Error] Failed to load checkpoint weights from '{self.checkpoint_path}': {e}")
                print(f"[Evaluator Warning] Could not load checkpoint ({e}). Initializing random test architecture as allow_untrained=True.")

        elif self.checkpoint_path and not os.path.exists(self.checkpoint_path):
            if not self.allow_untrained:
                raise FileNotFoundError(
                    f"[Evaluator Error] Checkpoint file '{self.checkpoint_path}' does not exist. "
                    f"To evaluate a trained model, provide a valid checkpoint. (Set allow_untrained=True for smoke tests only)."
                )
            print(f"[Evaluator Warning] Checkpoint '{self.checkpoint_path}' not found. Initializing untrained model for testing.")

        if not loaded_checkpoint:
            self.model = FinalTrajectoryModel(
                best_temporal_idx=self.best_ops[0],
                best_interaction_idx=self.best_ops[1],
                best_map_idx=self.best_ops[2],
                d_model=self.model_cfg.d_model,
                past_len=self.model_cfg.past_len,
                future_len=self.model_cfg.future_len,
                num_modes=getattr(self.model_cfg, "num_modes", 6),
                max_agents=getattr(self.model_cfg, "max_agents", 16),
                max_lanes=getattr(self.model_cfg, "max_lanes", 20),
                lane_pts=getattr(self.model_cfg, "lane_pts", 10)
            ).to(self.device)

        self.model.eval()

    def evaluate_dataset(
        self,
        batch_size: int = 16,
        max_batches: Optional[int] = None,
        custom_data_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs comprehensive multi-modal evaluation over the validation set.
        """
        ds = self.dataset_name or "nuscenes"
        bs = batch_size or 16
        _, val_loader = get_dataloaders(
            val_dir=custom_data_path,
            batch_size=bs,
            dataset_name=ds
        )

        total_samples = 0
        ade_1s_accum = 0.0
        ade_2s_accum = 0.0
        ade_3s_accum = 0.0
        fde_accum = 0.0
        mr_accum = 0.0
        step_errors_accum = torch.zeros(self.model_cfg.future_len, device=self.device)

        with torch.inference_mode():
            for batch_idx, batch in enumerate(val_loader):
                if max_batches and batch_idx >= max_batches:
                    break

                past = batch["past"].to(self.device)
                future = batch["future"].to(self.device)
                if future.dim() == 4:
                    future = future[:, :, 0, :]

                agents_past = batch.get("agents_past", None)
                if agents_past is not None and hasattr(agents_past, "to"):
                    agents_past = agents_past.to(self.device)

                map_lanes = batch.get("map_lanes", None)
                if map_lanes is not None and hasattr(map_lanes, "to"):
                    map_lanes = map_lanes.to(self.device)

                b_size = past.size(0)
                pred_trajs, mode_logits = self.model(past, agents_past=agents_past, map_lanes=map_lanes)

                # Top-K multi-modal evaluation (K=6)
                ade_1s_accum += min_ade_k(pred_trajs[:, :, :10, :], future[:, :10, :], k=6).item() * b_size
                ade_2s_accum += min_ade_k(pred_trajs[:, :, :20, :], future[:, :20, :], k=6).item() * b_size
                ade_3s_accum += min_ade_k(pred_trajs, future, k=6).item() * b_size
                fde_accum += min_fde_k(pred_trajs, future, k=6).item() * b_size
                mr_accum += miss_rate_k(pred_trajs, future, k=6, threshold=2.0) * b_size

                # Horizon step errors along winning mode
                target_exp = future.unsqueeze(1)
                step_dists = torch.norm(pred_trajs - target_exp, p=2, dim=-1) # [B, K, T]
                min_step_dists, _ = torch.min(step_dists, dim=1) # [B, T]
                step_errors_accum += torch.sum(min_step_dists, dim=0)

                total_samples += b_size

        n_samples = max(1, total_samples)
        step_errors = (step_errors_accum / n_samples).cpu().numpy().tolist()

        return {
            "dataset": self.dataset_name,
            "samples_evaluated": n_samples,
            "min_ade_1s": float(round(ade_1s_accum / n_samples, 4)),
            "min_ade_2s": float(round(ade_2s_accum / n_samples, 4)),
            "min_ade_3s": float(round(ade_3s_accum / n_samples, 4)),
            "min_fde_3s": float(round(fde_accum / n_samples, 4)),
            "miss_rate_6_2m": float(round((mr_accum / n_samples) * 100.0, 2)),
            "step_errors": [round(e, 4) for e in step_errors]
        }

    def profile_hardware(self, num_warmup: int = 25, num_runs: int = 100) -> Dict[str, Any]:
        """Profiles latency, FPS, and model complexity via HardwareProfiler."""
        return self.profiler.profile(num_warmup=num_warmup, num_runs=num_runs)

    def print_evaluation_report(self, eval_results: Dict[str, Any], hw_results: Dict[str, Any]):
        """Prints academic performance report table."""
        print("=" * 80)
        print(" TRAJECTORY LNAS: COMPREHENSIVE PERFORMANCE & HARDWARE AUDIT")
        print("=" * 80)
        print(f" Dataset Target:       {eval_results['dataset'].upper()}")
        print(f" Samples Evaluated:    {eval_results['samples_evaluated']:,}")
        print(f" Hardware Device:      {hw_results['device']}")
        print(f" Selected Temporal:    {hw_results['operators']['temporal']}")
        print(f" Selected Interaction: {hw_results['operators']['interaction']}")
        print(f" Selected HD Map:      {hw_results['operators']['map']}")
        print("-" * 80)
        print(" MULTI-MODAL TRAJECTORY ACCURACY METRICS (K=6 Hypotheses):")
        print(f"   • minADE @ 1.0s (10 timesteps): {eval_results['min_ade_1s']:.3f} meters")
        print(f"   • minADE @ 2.0s (20 timesteps): {eval_results['min_ade_2s']:.3f} meters")
        print(f"   • minADE @ 3.0s (30 timesteps): {eval_results['min_ade_3s']:.3f} meters")
        print(f"   • minFDE @ 3.0s (Final Step):   {eval_results['min_fde_3s']:.3f} meters")
        print(f"   • Miss Rate@6 (MR @ 2.0m):      {eval_results['miss_rate_6_2m']:.2f}%")
        print("-" * 80)
        print(" EMBEDDED HARDWARE PROFILING:")
        print(f"   • Model Parameters:          {hw_results['total_params']:,} ({hw_results['total_params_m']} M)")
        print(f"   • Theoretical FLOPs:         {hw_results['theoretical_flops_g']} GFLOPs")
        print(f"   • Inference Latency:         {hw_results['latency_ms_mean']:.2f} ± {hw_results['latency_ms_std']:.2f} ms")
        print(f"   • 95th Percentile Latency:   {hw_results['latency_ms_p95']:.2f} ms")
        print(f"   • Throughput / Frame Rate:   {hw_results['throughput_fps']:.1f} FPS")

    @staticmethod
    def print_sota_comparison(eval_metrics: Optional[Dict[str, Any]] = None, hw_metrics: Optional[Dict[str, Any]] = None, search_runtime_seconds: Optional[float] = None):
        """Prints benchmark comparison against published literature baselines."""
        print("=" * 80)
        print(" SOTA BENCHMARK COMPARISON ON ARGOVERSE 1 & NUSCENES FORECASTING (3.0s Horizon)")
        print("=" * 80)
        print(f"{'Method / Model':<26} | {'minADE (m)':<10} | {'minFDE (m)':<10} | {'Params (M)':<10} | {'FLOPs (G)':<9} | {'Search Cost':<18}")
        print("-" * 80)

        baselines = [
            ("Vanilla LSTM [Baseline]", 1.74, 3.82, 0.45, 0.38, "Manual Design"),
            ("Trajectron++ [ECCV 2020]", 1.02, 2.18, 1.28, 1.42, "Manual Design"),
            ("LaneGCN [ECCV 2020]", 0.87, 1.79, 2.15, 2.85, "Manual Design"),
            ("TrajectoryNAS [ICRA 2022]", 0.85, 1.73, 1.94, 2.10, "120 GPU-h (MOSA)")
        ]

        for name, ade_v, fde_v, par_v, fl_v, cost_v in baselines:
            print(f"{name:<26} | {ade_v:<10.2f} | {fde_v:<10.2f} | {par_v:<10.2f} | {fl_v:<9.2f} | {cost_v:<18}")

        if eval_metrics and hw_metrics:
            actual_ade = eval_metrics.get("min_ade_3s", "N/A")
            actual_fde = eval_metrics.get("min_fde_3s", "N/A")
            actual_params = hw_metrics.get("total_params_m", "N/A")
            actual_flops = hw_metrics.get("theoretical_flops_g", "N/A")
            
            if search_runtime_seconds is not None:
                h = search_runtime_seconds / 3600.0
                actual_cost = f"{h:.2f} GPU-h (DARTS)" if h >= 0.1 else f"{search_runtime_seconds:.0f}s (DARTS)"
            else:
                actual_cost = "N/A"

            curr_label = "TrajectoryLNAS [This Run] [*]"
            print(f"{curr_label:<26} | {str(actual_ade):<10} | {str(actual_fde):<10} | {str(actual_params):<10} | {str(actual_flops):<9} | {actual_cost:<18}")

        print("=" * 80)


def run_evaluation_cli():
    """CLI runner function for python -m evaluate.evaluator"""
    parser = argparse.ArgumentParser(description="TrajectoryLNAS Evaluation & Benchmarking Suite")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_trajectory_lnas.pt")
    parser.add_argument("--dataset", type=str, default="nuscenes", choices=["nuscenes", "argoverse"])
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--report_dir", type=str, default="report")
    parser.add_argument("--plot", action="store_true", default=True, help="Save evaluation plots")
    parser.add_argument("--allow_untrained", action="store_true", default=False, help="Allow testing with random weights")
    args = parser.parse_args()

    os.makedirs(args.report_dir, exist_ok=True)
    plots_dir = os.path.join(args.report_dir, "plots")

    evaluator = TrajectoryEvaluator(
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        device=args.device,
        allow_untrained=args.allow_untrained
    )

    eval_results = evaluator.evaluate_dataset(
        batch_size=args.batch_size,
        custom_data_path=args.data_path
    )

    hw_results = evaluator.profile_hardware()

    evaluator.print_evaluation_report(eval_results, hw_results)
    evaluator.print_sota_comparison(eval_results, hw_results)

    if args.plot:
        plot_horizon_error_curve(eval_results.get("step_errors", []), dataset=args.dataset, output_dir=plots_dir)
        plot_benchmark_comparison(dataset=args.dataset, output_dir=plots_dir)

    json_path = os.path.join(args.report_dir, f"evaluation_results_{args.dataset}.json")
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "thesis": {
            "project": "TrajectoryLNAS",
            "student": "Alireza Ghafouri (4031631209)",
            "supervisors": ["Dr. Zeinab Ghasemi Naraaghi", "Dr. Zeinab Abedini"]
        },
        "evaluation": eval_results,
        "hardware": hw_results
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[Report] Full evaluation metrics saved to: {json_path}")


if __name__ == "__main__":
    run_evaluation_cli()
