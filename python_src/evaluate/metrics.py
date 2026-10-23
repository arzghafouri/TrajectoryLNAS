"""
Trajectory Forecasting Accuracy & Safety Evaluation Suite
==========================================================
Implements standard regression and safety benchmarks for trajectory prediction:
  1. Geometric Displacement Accuracy (Regression): ADE, FDE, Step Errors
  2. Multi-Modal Trajectory Evaluation: minADE@K, minFDE@K, Miss Rate@K (threshold = 2.0m)
  3. Physical Safety & Collision Assessment: Collision Rate, Safety Margins
"""

from typing import Dict, Any, Optional
import torch


# ==============================================================================
# 1. Geometric Accuracy Metrics (Single-Trajectory & Multi-Modal Routing)
# ==============================================================================

def ade(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes Average Displacement Error (ADE). Auto-routes 4D multi-modal predictions to minADE@6."""
    if pred.dim() == 4:
        return min_ade_k(pred, target, k=6)
    dist = torch.norm(pred - target, p=2, dim=-1)
    return torch.mean(dist)


def fde(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes Final Displacement Error (FDE). Auto-routes 4D multi-modal predictions to minFDE@6."""
    if pred.dim() == 4:
        return min_fde_k(pred, target, k=6)
    dist = torch.norm(pred[:, -1, :] - target[:, -1, :], p=2, dim=-1)
    return torch.mean(dist)


def miss_rate(pred: torch.Tensor, target: torch.Tensor, threshold: float = 2.0) -> float:
    """Computes Miss Rate (MR @ threshold). Auto-routes 4D multi-modal predictions to MissRate@6."""
    if pred.dim() == 4:
        return miss_rate_k(pred, target, k=6, threshold=threshold)
    dist = torch.norm(pred[:, -1, :] - target[:, -1, :], p=2, dim=-1)
    return torch.mean((dist > threshold).float()).item()


def best_of_k_loss(pred_trajs: torch.Tensor, target: torch.Tensor, mode_logits: Optional[torch.Tensor] = None):
    """
    Winner-Takes-All (WTA) multi-modal loss:
      1. Finds best trajectory mode k* minimizing ADE + 0.5 * FDE to ground truth.
      2. Regression loss on the winning trajectory k*.
      3. Cross-entropy classification loss pulling logits towards winning mode k*.
    """
    if pred_trajs.dim() == 3:
        loss_reg = ade(pred_trajs, target) + 0.5 * fde(pred_trajs, target)
        loss_cls = torch.tensor(0.0, device=pred_trajs.device)
        return loss_reg, loss_cls

    B, K, T, _ = pred_trajs.shape
    target_exp = target.unsqueeze(1) # [B, 1, T, 2]
    step_errors = torch.norm(pred_trajs - target_exp, p=2, dim=-1).mean(dim=-1) # [B, K] ADE per mode
    end_errors = torch.norm(pred_trajs[:, :, -1, :] - target_exp[:, :, -1, :], p=2, dim=-1) # [B, K] FDE per mode
    total_mode_errors = step_errors + 0.5 * end_errors # [B, K]

    best_mode_idx = torch.argmin(total_mode_errors, dim=-1) # [B]
    batch_idx = torch.arange(B, device=pred_trajs.device)
    winning_trajs = pred_trajs[batch_idx, best_mode_idx] # [B, T, 2]

    loss_reg = ade(winning_trajs, target) + 0.5 * fde(winning_trajs, target)
    if mode_logits is not None:
        loss_cls = torch.nn.functional.cross_entropy(mode_logits, best_mode_idx)
    else:
        loss_cls = torch.tensor(0.0, device=pred_trajs.device)

    return loss_reg, loss_cls


# ==============================================================================
# 2. Multi-Modal Forecasting Metrics (minADE@K, minFDE@K, MissRate@K)
# ==============================================================================

def min_ade_k(multi_preds: torch.Tensor, target: torch.Tensor, k: int = 6) -> torch.Tensor:
    """
    Computes minADE@K: minimum ADE across the top-K predicted trajectory modes.
    multi_preds: [Batch, K_modes, Horizon, 2]
    target: [Batch, Horizon, 2]
    """

    if multi_preds.dim() == 3:
        return ade(multi_preds, target)
    
    k_eval = min(k, multi_preds.shape[1])
    preds_k = multi_preds[:, :k_eval, :, :]  # [B, K, T, 2]
    target_exp = target.unsqueeze(1)         # [B, 1, T, 2]
    
    # Euclidean distance per step: [B, K, T]
    step_dists = torch.norm(preds_k - target_exp, p=2, dim=-1)
    mode_ades = torch.mean(step_dists, dim=-1) # [B, K]
    min_ades, _ = torch.min(mode_ades, dim=-1)  # [B]
    return torch.mean(min_ades)


def min_fde_k(multi_preds: torch.Tensor, target: torch.Tensor, k: int = 6) -> torch.Tensor:
    """
    Computes minFDE@K: minimum endpoint displacement error among top-K modes.
    multi_preds: [Batch, K_modes, Horizon, 2]
    target: [Batch, Horizon, 2]
    """
    if multi_preds.dim() == 3:
        return fde(multi_preds, target)
    
    k_eval = min(k, multi_preds.shape[1])
    end_preds = multi_preds[:, :k_eval, -1, :] # [B, K, 2]
    end_target = target[:, -1, :].unsqueeze(1) # [B, 1, 2]
    
    mode_fdes = torch.norm(end_preds - end_target, p=2, dim=-1) # [B, K]
    min_fdes, _ = torch.min(mode_fdes, dim=-1)                  # [B]
    return torch.mean(min_fdes)


def miss_rate_k(multi_preds: torch.Tensor, target: torch.Tensor, k: int = 6, threshold: float = 2.0) -> float:
    """
    Computes Miss Rate@K: percentage of scenarios where NONE of the K modes is within threshold.
    """
    if multi_preds.dim() == 3:
        return miss_rate(multi_preds, target, threshold)
    
    k_eval = min(k, multi_preds.shape[1])
    end_preds = multi_preds[:, :k_eval, -1, :] # [B, K, 2]
    end_target = target[:, -1, :].unsqueeze(1) # [B, 1, 2]
    
    mode_fdes = torch.norm(end_preds - end_target, p=2, dim=-1) # [B, K]
    min_fdes, _ = torch.min(mode_fdes, dim=-1)                  # [B]
    return torch.mean((min_fdes > threshold).float()).item()


# ==============================================================================
# 3. Physical Safety & Collision Risk (Collision Rate & Safety Margin)
# ==============================================================================

def evaluate_collision_risk(pred: torch.Tensor, neighbors: Optional[torch.Tensor] = None, vehicle_radius: float = 1.8) -> Dict[str, Any]:
    """
    Computes Collision Rate (CR) against surrounding dynamic agents and physical safety envelopes.
    Standard safety metric in autonomous driving prediction challenges.
    """
    batch_size, horizon, _ = pred.shape
    collision_counts = 0
    safe_counts = 0
    
    if neighbors is not None and neighbors.dim() >= 3:
        for t in range(horizon):
            p_pos = pred[:, t, :] # [B, 2]
            n_pos = neighbors[:, 0, min(t, neighbors.shape[2]-1), :] # Closest agent
            dist = torch.norm(p_pos - n_pos, p=2, dim=-1)
            
            critical = (dist < vehicle_radius).float().sum().item()
            if critical > 0:
                collision_counts += 1
            else:
                safe_counts += 1
    else:
        # Kinematic lateral acceleration check as safety indicator
        step_accel = torch.norm(torch.diff(pred, n=2, dim=1), p=2, dim=-1)
        max_accels, _ = torch.max(step_accel, dim=-1)
        crit_mask = max_accels > 4.5
        collision_counts = int(crit_mask.sum().item())

    collision_rate = (collision_counts / max(1, batch_size)) * 100.0
    return {
        "collision_rate_percent": round(collision_rate, 2),
        "safety_compliance_percent": round(100.0 - collision_rate, 2)
    }


