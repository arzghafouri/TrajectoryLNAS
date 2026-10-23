"""
Hardware Profiler & Latency Benchmark Engine
============================================
Single Responsibility Principle (SRP):
Encapsulates runtime profiling (latency, throughput, P95, parameter count)
and architectural FLOPs / complexity modeling for both SuperNets and standalone models.
Benchmarking strictly runs under torch.inference_mode() with CUDA synchronization.
"""

import time
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import numpy as np


def get_supernet_costs(alphas_dict: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], all_costs: Dict[str, Any]) -> torch.Tensor:
    """
    Computes expected FLOPs of the SuperNet under current architectural distribution alpha.
    Used as differentiable hardware penalty in total loss:
      L_total = L_ADE + L_FDE + lambda * expected_FLOPs
    """
    a_temp, a_inter, a_map = alphas_dict
    
    flops_temp = sum(w * c for w, c in zip(a_temp, all_costs["temporal"]))
    flops_inter = sum(w * c for w, c in zip(a_inter, all_costs["interaction"]))
    flops_map = sum(w * c for w, c in zip(a_map, all_costs["map"]))
    
    # Base invariant cost
    total_flops = flops_temp + flops_inter + flops_map + all_costs.get("base", 0.0)
    
    # Scale to GigaFLOPs (1e9) for numerical balance with ADE/FDE loss
    return total_flops / 1e9


def get_supernet_flops(alphas_dict: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], all_costs: Dict[str, Any]) -> torch.Tensor:
    """Alias for get_supernet_costs."""
    return get_supernet_costs(alphas_dict, all_costs)


def get_supernet_complexity(alphas_dict: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], all_costs: Dict[str, Any]) -> torch.Tensor:
    """Alias for get_supernet_costs."""
    return get_supernet_costs(alphas_dict, all_costs)


def simple_complexity_proxy(alphas_dict: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Simpler proxy based purely on parameter sums."""
    a_temp, a_inter, a_map = alphas_dict
    return torch.sum(a_temp) + torch.sum(a_inter) + torch.sum(a_map)


class HardwareProfiler:
    """Profiles execution latency, throughput (FPS), and model footprint under strict inference mode."""
    def __init__(self, model: nn.Module, device: torch.device, past_len: int = 20, max_agents: int = 16, max_lanes: int = 20, lane_pts: int = 10):
        self.model = model
        self.device = device
        self.past_len = past_len
        self.max_agents = max_agents
        self.max_lanes = max_lanes
        self.lane_pts = lane_pts

    def profile(self, num_warmup: int = 25, num_runs: int = 100) -> Dict[str, Any]:
        self.model.eval()
        dummy_past = torch.randn(1, self.past_len, 2, device=self.device)
        dummy_agents = torch.randn(1, self.past_len, self.max_agents, 2, device=self.device)
        dummy_lanes = torch.randn(1, self.max_lanes, self.lane_pts, 2, device=self.device)

        with torch.inference_mode():
            # Warm-up phase
            for _ in range(num_warmup):
                _ = self.model(dummy_past, agents_past=dummy_agents, map_lanes=dummy_lanes)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            # Latency benchmark
            latencies = []
            for _ in range(num_runs):
                t0 = time.perf_counter()
                _ = self.model(dummy_past, agents_past=dummy_agents, map_lanes=dummy_lanes)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000.0)

        lat_arr = np.array(latencies)
        mean_lat = float(np.mean(lat_arr))
        std_lat = float(np.std(lat_arr))
        p95_lat = float(np.percentile(lat_arr, 95))
        throughput_fps = float(1000.0 / max(1e-4, mean_lat))

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        summary = getattr(self.model, "get_summary", lambda: {})()

        # Extract analytical FLOPs without arbitrary fallbacks
        flops_val = summary.get("theoretical_flops_g", None)

        return {
            "device": str(self.device),
            "latency_ms_mean": round(mean_lat, 2),
            "latency_ms_std": round(std_lat, 2),
            "latency_ms_p95": round(p95_lat, 2),
            "throughput_fps": round(throughput_fps, 1),
            "total_params": total_params,
            "total_params_m": round(total_params / 1e6, 3),
            "trainable_params": trainable_params,
            "theoretical_flops_g": flops_val,
            "operators": {
                "temporal": summary.get("temporal_op", "N/A"),
                "interaction": summary.get("interaction_op", "N/A"),
                "map": summary.get("map_op", "N/A")
            }
        }
