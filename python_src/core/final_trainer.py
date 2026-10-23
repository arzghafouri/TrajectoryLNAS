"""
Standalone Final Model Trainer
==============================
Trains the discretized FinalTrajectoryModel discovered by TrajectoryLNAS.
Implements multi-agent interactions, HD maps, multi-modal loss, AMP FP16, and Early Stopping.
"""

import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Any, Tuple

from core.base_trainer import BaseTrainer
from config.search_config import TrainConfig, ModelConfig
from models.final_model import FinalTrajectoryModel
from utils import get_dataloaders, set_seed
from evaluate.metrics import min_ade_k, min_fde_k, miss_rate_k, best_of_k_loss

class FinalModelTrainer(BaseTrainer):
    """
    Object-Oriented Standalone Trainer for the Discovered Architecture.
    """
    def __init__(
        self,
        best_op_indices: Tuple[int, int, int],
        train_config: TrainConfig,
        model_config: ModelConfig = None
    ):
        super().__init__(device=train_config.device)
        self.cfg = train_config
        self.model_cfg = model_config or ModelConfig()
        self.best_t, self.best_i, self.best_m = [int(x) for x in best_op_indices]

        # Seed fixing for reproducible standalone retraining
        set_seed(getattr(self.cfg, "seed", 42))

        # 1. Instantiate Discretized Model
        self.model = FinalTrajectoryModel(
            best_temporal_idx=self.best_t,
            best_interaction_idx=self.best_i,
            best_map_idx=self.best_m,
            d_model=self.model_cfg.d_model,
            past_len=self.model_cfg.past_len,
            future_len=self.model_cfg.future_len,
            num_modes=getattr(self.model_cfg, "num_modes", 6),
            max_agents=getattr(self.model_cfg, "max_agents", 16),
            max_lanes=getattr(self.model_cfg, "max_lanes", 20),
            lane_pts=getattr(self.model_cfg, "lane_pts", 10)
        ).to(self.device)

        # 2. Optimizer & Cosine Annealing Scheduler
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg.epochs,
            eta_min=1e-5
        )

        # 3. DataLoaders
        train_pct = getattr(self.cfg, "train_samples_percent", None)
        self.train_loader, self.val_loader = get_dataloaders(
            train_path=self.cfg.train_path,
            val_path=self.cfg.val_path,
            batch_size=self.cfg.batch_size,
            dataset_type=self.cfg.dataset,
            sample_percent=train_pct
        )

        self.best_val_ade = float("inf")
        self.best_val_fde = float("inf")

        # AMP and Gradient Accumulation for 4GB VRAM
        self.use_amp = getattr(self.cfg, "use_amp", True) and (self.device.type == "cuda")
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.grad_accum_steps = max(1, getattr(self.cfg, "grad_accum_steps", 8))

    def train_epoch(self, epoch: int) -> float:
        """Trains for one full epoch with AMP and Gradient Accumulation."""
        self.model.train()
        total_loss = 0.0
        step_count = 0
        self.optimizer.zero_grad()

        num_batches = len(self.train_loader)
        accum_counter = 0

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

            try:
                autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=self.use_amp)
            except (AttributeError, TypeError):
                autocast_ctx = torch.cuda.amp.autocast(enabled=self.use_amp)

            with autocast_ctx:
                pred_trajs, mode_logits = self.model(past, agents_past=agents_past, map_lanes=map_lanes)
                loss_reg, loss_cls = best_of_k_loss(pred_trajs, future, mode_logits)

                # Physics-Informed Kinematic Regularization on predicted modes
                diff2 = pred_trajs[:, :, 2:, :] - 2 * pred_trajs[:, :, 1:-1, :] + pred_trajs[:, :, :-2, :]
                loss_physics = torch.mean(torch.norm(diff2, p=2, dim=-1))

                # Combined Multi-Modal Objective
                loss_batch = loss_reg + (0.1 * loss_cls) + (0.05 * loss_physics)
                loss_scaled = loss_batch / self.grad_accum_steps

            self.scaler.scale(loss_scaled).backward()
            accum_counter += 1

            if accum_counter % self.grad_accum_steps == 0 or (batch_idx + 1) == num_batches:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                accum_counter = 0

            total_loss += loss_batch.item()
            step_count += 1

        if step_count > 0:
            self.scheduler.step()

        return total_loss / max(1, step_count)

    def validate(self, epoch: int) -> Dict[str, float]:
        """Evaluates model performance on validation set using exact sample weighting."""
        self.model.eval()
        sum_ade = 0.0
        sum_fde = 0.0
        sum_mr = 0.0
        total_samples = 0

        with torch.inference_mode():
            for batch in self.val_loader:
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

                pred_trajs, _ = self.model(past, agents_past=agents_past, map_lanes=map_lanes)
                B = past.shape[0]
                total_samples += B

                sum_ade += min_ade_k(pred_trajs, future, k=6).item() * B
                sum_fde += min_fde_k(pred_trajs, future, k=6).item() * B
                sum_mr += miss_rate_k(pred_trajs, future, k=6, threshold=2.0) * B

        val_samples = max(1, total_samples)
        mean_ade = sum_ade / val_samples
        mean_fde = sum_fde / val_samples
        mean_mr = (sum_mr / val_samples) * 100.0

        return {
            "val_ade": mean_ade,
            "val_fde": mean_fde,
            "miss_rate": mean_mr
        }

    def train_and_evaluate(self) -> FinalTrajectoryModel:
        """Runs the complete training loop with Early Stopping and best weight restoration."""
        summary = self.model.get_summary()
        print("=" * 80)
        print(" Standalone Training of Discovered TrajectoryLNAS Architecture")
        print(f" Target Dataset:       {self.cfg.dataset.upper()}")
        print(f" Temporal Operator:    {summary['temporal_op']} (idx {summary['best_temporal_idx']})")
        print(f" Interaction Operator: {summary['interaction_op']} (idx {summary['best_interaction_idx']})")
        print(f" HD Map Operator:      {summary['map_op']} (idx {summary['best_map_idx']})")
        print(f" Modes (K):            {summary['num_modes']}")
        print(f" Total Parameters:     {summary['parameters']:,}")
        print(f" Theoretical FLOPs:    {summary['theoretical_flops_g']} GFLOPs")
        print(f" Early Stopping:       Patience = {self.cfg.early_stopping_patience} epochs | Min Delta = {self.cfg.early_stopping_min_delta}m")
        print("=" * 80)

        self.training_history = []
        patience_counter = 0
        best_epoch = 0

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self.train_epoch(epoch)
            metrics = self.validate(epoch)

            mean_ade = metrics["val_ade"]
            mean_fde = metrics["val_fde"]
            mean_mr = metrics["miss_rate"]

            improvement = (self.best_val_ade - mean_ade)
            is_best = improvement > self.cfg.early_stopping_min_delta

            if is_best:
                self.best_val_ade = mean_ade
                self.best_val_fde = mean_fde
                best_epoch = epoch
                patience_counter = 0
                self.save_checkpoint({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "best_ade": self.best_val_ade,
                    "best_fde": self.best_val_fde,
                    "config": summary
                }, self.cfg.save_path)
                flag = " [BEST CHECKPOINT SAVED]"
            else:
                patience_counter += 1
                flag = f" (Patience: {patience_counter}/{self.cfg.early_stopping_patience})"

            self.training_history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_ade": round(mean_ade, 4),
                "val_fde": round(mean_fde, 4),
                "miss_rate_2m": round(mean_mr, 2),
                "is_best": is_best
            })

            print(
                f"Epoch [{epoch:02d}/{self.cfg.epochs:02d}] "
                f"Train Loss: {train_loss:.3f} | "
                f"Val minADE@6: {mean_ade:.3f}m | "
                f"Val minFDE@6: {mean_fde:.3f}m | "
                f"Miss Rate@6: {mean_mr:.1f}%{flag}"
            )

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            if patience_counter >= self.cfg.early_stopping_patience:
                print(f"\n[Early Stopping Triggered] No validation ADE improvement for {self.cfg.early_stopping_patience} consecutive epochs.")
                print(f"  -> Early stopping activated at epoch {epoch}. Restoring optimal weights from epoch {best_epoch}...")
                break

        # Reload best model weights
        if os.path.exists(self.cfg.save_path):
            try:
                try:
                    ckpt = torch.load(self.cfg.save_path, map_location=self.device, weights_only=False)
                except TypeError:
                    ckpt = torch.load(self.cfg.save_path, map_location=self.device)
                self.model.load_state_dict(ckpt["model_state_dict"])
                print(f"[Model Restored] Successfully reloaded optimal weights from epoch {best_epoch} (Val minADE@6: {self.best_val_ade:.3f}m)")
            except Exception as e:
                print(f"[Model Restore Warning] Could not reload checkpoint ({e}).")

        print("=" * 80)
        print(" Final Model Training Complete.")
        print(f" Best Validation minADE@6: {self.best_val_ade:.3f} meters (Epoch {best_epoch})")
        print(f" Best Validation minFDE@6: {self.best_val_fde:.3f} meters")
        print(f" Model Weights Saved: {self.cfg.save_path}")
        print("=" * 80)

        return self.model
