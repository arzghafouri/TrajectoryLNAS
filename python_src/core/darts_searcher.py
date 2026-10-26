"""
Differentiable Neural Architecture Searcher (DARTS)
===================================================
Encapsulates bi-level optimization for continuous SuperNet search with hardware FLOPs penalty.
Supports multi-agent interactions, vectorized HD maps, and multi-modal trajectory forecasting.
"""

import math
import time
import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Dict, Any, Tuple

# Ensure python_src directory is in sys.path
_curr_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_curr_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from core.base_trainer import BaseTrainer
from config.search_config import SearchConfig, ModelConfig
from models.search_supernet import SearchSuperNet, ArchParams
from utils import get_dataloaders, set_seed
from evaluate.metrics import ade, fde, min_ade_k, min_fde_k, best_of_k_loss
from evaluate.hardware_profiler import get_supernet_costs
from evaluate.zero_cost_proxies import snip_score, synflow_score

class DARTSSearcher(BaseTrainer):
    """
    Object-Oriented DARTS Search Engine for Trajectory Prediction.
    
    Solves the bi-level optimization problem:
      min_alpha L_val(w*(alpha), alpha) + lambda * FLOPs(alpha)
      s.t. w*(alpha) = argmin_w L_train(w, alpha)
    """
    def __init__(self, search_config: SearchConfig, model_config: ModelConfig = None):
        super().__init__(device=search_config.device)
        self.cfg = search_config
        self.model_cfg = model_config or ModelConfig()

        # Seed fixing for complete reproducibility across runs
        set_seed(getattr(self.cfg, "seed", 42))

        # 1. Initialize Networks
        self.supernet = SearchSuperNet(
            d_model=self.model_cfg.d_model,
            past_len=self.model_cfg.past_len,
            future_len=self.model_cfg.future_len,
            num_modes=getattr(self.model_cfg, "num_modes", 6),
            max_agents=getattr(self.model_cfg, "max_agents", 16),
            max_lanes=getattr(self.model_cfg, "max_lanes", 20),
            lane_pts=getattr(self.model_cfg, "lane_pts", 10)
        ).to(self.device)

        self.arch_params = ArchParams().to(self.device)

        # 2. Distinct Optimizers for Weights (w) and Architecture (alpha)
        self.opt_w = Adam(
            self.supernet.parameters(),
            lr=self.cfg.lr_weights,
            weight_decay=self.cfg.weight_decay
        )
        self.opt_a = Adam(
            self.arch_params.parameters(),
            lr=self.cfg.lr_alphas,
            betas=(0.5, 0.999),
            weight_decay=self.cfg.alpha_weight_decay
        )

        # 3. DataLoaders
        search_pct = getattr(self.cfg, "search_samples_percent", 20.0)
        self.train_loader, self.val_loader = get_dataloaders(
            train_path=self.cfg.train_path,
            val_path=self.cfg.val_path,
            batch_size=self.cfg.batch_size,
            dataset_type=self.cfg.dataset,
            sample_percent=search_pct
        )
        self.val_iter = None
        self.all_costs = self.supernet.get_ops_flops()
        self.elapsed_seconds = 0.0

        # Temperature schedule parameters for annealing
        self.temp_ratio = (self.cfg.temperature_min / self.cfg.temperature_start) ** (1.0 / max(1, self.cfg.epochs))

        # Mixed Precision (AMP) & Gradient Accumulation for 4GB VRAM
        self.use_amp = getattr(self.cfg, "use_amp", True) and (self.device.type == "cuda")
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.grad_accum_steps = max(1, getattr(self.cfg, "grad_accum_steps", 16))

    def _get_current_temperature(self, epoch: int) -> float:
        """Anneals Gumbel/Softmax temperature to encourage sharp discrete selection."""
        return max(self.cfg.temperature_min, self.cfg.temperature_start * (self.temp_ratio ** epoch))

    def profile_zero_cost_proxies(self) -> Dict[str, float]:
        """Calculates pre-training gradient and saliency flow proxies."""
        sample_batch = next(iter(self.val_loader))
        snip = snip_score(self.supernet, sample_batch, self.device)
        synflow = synflow_score(self.supernet, self.device)
        return {"snip": snip, "synflow": synflow}

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Inner loop: Updates network weights (w) with Gradient Accumulation & AMP."""
        self.supernet.train()
        total_w_loss = 0.0
        step_count = 0
        temperature = self._get_current_temperature(epoch)
        self.opt_w.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
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

            category_id = batch.get("category_id", None)
            if category_id is not None and hasattr(category_id, "to"):
                category_id = category_id.to(self.device)

            map_lane_types = batch.get("map_lane_types", None)
            if map_lane_types is not None and hasattr(map_lane_types, "to"):
                map_lane_types = map_lane_types.to(self.device)

            neighbor_categories = batch.get("neighbor_categories", None)
            if neighbor_categories is not None and hasattr(neighbor_categories, "to"):
                neighbor_categories = neighbor_categories.to(self.device)

            alphas_dict = self.arch_params.softmax_all(temperature=temperature)

            try:
                autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=self.use_amp)
            except (AttributeError, TypeError):
                autocast_ctx = torch.cuda.amp.autocast(enabled=self.use_amp)

            with autocast_ctx:
                pred_trajs, mode_logits = self.supernet(
                    past, alphas_dict, agents_past=agents_past, map_lanes=map_lanes,
                    category_id=category_id, map_lane_types=map_lane_types,
                    neighbor_categories=neighbor_categories
                )
                loss_reg, loss_cls = best_of_k_loss(pred_trajs, future, mode_logits)
                loss_w = (loss_reg + 0.1 * loss_cls) / self.grad_accum_steps

            self.scaler.scale(loss_w).backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                self.scaler.unscale_(self.opt_w)
                torch.nn.utils.clip_grad_norm_(self.supernet.parameters(), max_norm=5.0)
                self.scaler.step(self.opt_w)
                self.scaler.update()
                self.opt_w.zero_grad()

            total_w_loss += (loss_w.item() * self.grad_accum_steps)
            step_count += 1

        avg_loss = total_w_loss / max(1, step_count)
        return {"train_w_loss": avg_loss, "step_count": step_count}

    def validate(self, epoch: int, is_warmup: bool = False) -> Dict[str, Any]:
        """
        Outer loop: Updates architectural parameters (alpha) across validation batches.
        During warmup (is_warmup=True), architectural parameters are frozen.
        """
        self.supernet.train()
        temperature = self._get_current_temperature(epoch)
        n_val_batches = getattr(self.cfg, "val_batches_for_alpha", 4)

        val_ades = []
        val_fdes = []

        if not is_warmup:
            self.opt_a.zero_grad()

        total_arch_loss = 0.0

        for _ in range(n_val_batches):
            try:
                val_batch = next(self.val_iter)
            except (TypeError, StopIteration):
                self.val_iter = iter(self.val_loader)
                val_batch = next(self.val_iter)

            past = val_batch["past"].to(self.device)
            future = val_batch["future"].to(self.device)
            if future.dim() == 4:
                future = future[:, :, 0, :]

            agents_past = val_batch.get("agents_past", None)
            if agents_past is not None and hasattr(agents_past, "to"):
                agents_past = agents_past.to(self.device)

            map_lanes = val_batch.get("map_lanes", None)
            if map_lanes is not None and hasattr(map_lanes, "to"):
                map_lanes = map_lanes.to(self.device)

            category_id = val_batch.get("category_id", None)
            if category_id is not None and hasattr(category_id, "to"):
                category_id = category_id.to(self.device)

            map_lane_types = val_batch.get("map_lane_types", None)
            if map_lane_types is not None and hasattr(map_lane_types, "to"):
                map_lane_types = map_lane_types.to(self.device)

            neighbor_categories = val_batch.get("neighbor_categories", None)
            if neighbor_categories is not None and hasattr(neighbor_categories, "to"):
                neighbor_categories = neighbor_categories.to(self.device)

            alphas_dict = self.arch_params.softmax_all(temperature=temperature)

            with torch.set_grad_enabled(not is_warmup):
                pred_trajs, mode_logits = self.supernet(
                    past, alphas_dict, agents_past=agents_past, map_lanes=map_lanes,
                    category_id=category_id, map_lane_types=map_lane_types,
                    neighbor_categories=neighbor_categories
                )
                val_loss_ade = min_ade_k(pred_trajs, future, k=6)
                val_loss_fde = min_fde_k(pred_trajs, future, k=6)

                # Hardware FLOPs penalty
                flops_cost = get_supernet_costs(alphas_dict, self.all_costs)
                batch_arch_loss = val_loss_ade + val_loss_fde + (self.cfg.lambda_flops * flops_cost)

                if not is_warmup:
                    (batch_arch_loss / n_val_batches).backward()

                total_arch_loss += batch_arch_loss.item()
                val_ades.append(val_loss_ade.item())
                val_fdes.append(val_loss_fde.item())

        if not is_warmup:
            torch.nn.utils.clip_grad_norm_(self.arch_params.parameters(), max_norm=5.0)
            self.opt_a.step()

        a_temp, a_inter, a_map = alphas_dict
        mean_ade = float(sum(val_ades) / max(1, len(val_ades)))
        mean_fde = float(sum(val_fdes) / max(1, len(val_fdes)))

        return {
            "val_ade": mean_ade,
            "val_fde": mean_fde,
            "flops": flops_cost.item(),
            "temperature": temperature,
            "alphas": {
                "temporal": [round(x.item(), 4) for x in a_temp],
                "interaction": [round(x.item(), 4) for x in a_inter],
                "map": [round(x.item(), 4) for x in a_map]
            }
        }

    def run_search(self) -> Tuple[int, int, int]:
        """Runs complete bi-level DARTS search loop."""
        t_start = time.time()
        warmup_epochs = self.cfg.warmup_epochs

        print("=" * 80)
        print(" TrajectoryLNAS: Differentiable Neural Architecture Search (DARTS)")
        print(f" Dataset:        {self.cfg.dataset}")
        print(f" Device:         {self.device} (AMP={self.use_amp})")
        print(f" Batch Size:     {self.cfg.batch_size} (Grad Accum={self.grad_accum_steps})")
        print(f" Search Epochs:  {self.cfg.epochs} (Warmup={warmup_epochs})")
        print(f" FLOPs Penalty:  λ = {self.cfg.lambda_flops}")
        print("=" * 80)

        self.search_history = []
        self.proxies = self.profile_zero_cost_proxies()
        print(f"[Zero-Cost Proxies Initial] SNIP: {self.proxies['snip']:.4f} | SynFlow: {self.proxies['synflow']:.4f}")

        for epoch in range(1, self.cfg.epochs + 1):
            is_warmup = (epoch <= warmup_epochs)
            train_res = self.train_epoch(epoch)
            val_res = self.validate(epoch, is_warmup=is_warmup)

            record = {
                "epoch": epoch,
                "is_warmup": is_warmup,
                "train_loss": round(train_res["train_w_loss"], 4),
                "val_ade": round(val_res["val_ade"], 4),
                "val_fde": round(val_res["val_fde"], 4),
                "flops_g": round(val_res["flops"], 5),
                "temperature": round(val_res["temperature"], 3),
                "alphas": val_res["alphas"]
            }
            self.search_history.append(record)

            prefix = f"Epoch [{epoch:02d}/{self.cfg.epochs:02d}]"
            tag = f" [Warmup {epoch}/{warmup_epochs}]" if is_warmup else ""
            print(
                f"{prefix}{tag} "
                f"Train Loss: {train_res['train_w_loss']:.3f} | "
                f"Val minADE@6: {val_res['val_ade']:.3f}m | "
                f"Val minFDE@6: {val_res['val_fde']:.3f}m | "
                f"FLOPs: {val_res['flops']:.4f}G | "
                f"Temp: {val_res['temperature']:.2f}"
                + (" (Alpha Frozen)" if is_warmup else "")
            )
            print(f"  -> Temporal Alphas:    {val_res['alphas']['temporal']}")
            print(f"  -> Interaction Alphas: {val_res['alphas']['interaction']}")
            print(f"  -> Map Alphas:         {val_res['alphas']['map']}")

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        self.elapsed_seconds = time.time() - t_start
        print(f"\n[DARTSSearcher] Architecture search completed in {self.elapsed_seconds:.2f} seconds.")
        return self.discretize_architecture()

    def discretize_architecture(self) -> Tuple[int, int, int]:
        """Extracts the discrete winning operators via argmax(alpha)."""
        print("=" * 80)
        print(" DARTS Search Completed. Discretizing Optimal Architecture...")
        print("=" * 80)

        temporal_ops = ["Identity", "GRUEncoder", "LSTMEncoder", "TemporalConvEncoder", "TemporalSelfAttention"]
        interaction_ops = ["Identity", "GCNLayer", "GATLayer", "SocialPooling", "PairwiseAttention"]
        map_ops = ["Identity", "LaneAttention", "LaneGraphEncoder", "MapFeatureFusion"]

        with torch.no_grad():
            a_temp, a_inter, a_map = self.arch_params.softmax_all()
            best_t = torch.argmax(a_temp).item()
            best_i = torch.argmax(a_inter).item()
            best_m = torch.argmax(a_map).item()

        print(f"[*] Optimal Temporal Operator:    Index {best_t} -> {temporal_ops[best_t]}")
        print(f"[*] Optimal Interaction Operator: Index {best_i} -> {interaction_ops[best_i]}")
        print(f"[*] Optimal HD Map Operator:      Index {best_m} -> {map_ops[best_m]}")

        return best_t, best_i, best_m
