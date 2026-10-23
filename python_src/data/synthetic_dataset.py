"""
Synthetic Trajectory Dataset
============================
Single Responsibility Principle (SRP):
Dedicated exclusively to generating synthetic, kinematic-compliant multi-agent vehicle
trajectories for testing, validation, and zero-hardware fallback scenarios.
"""

import math
from typing import Dict, List
import torch
from torch.utils.data import Dataset


class SyntheticTrajectoryDataset(Dataset):
    """
    Generates realistic vehicle trajectory sequences replicating Argoverse/nuScenes benchmarks:
      - past: 20 timesteps (2.0 sec @ 10Hz)
      - future: 30 timesteps (3.0 sec @ 10Hz)
      - realistic vehicle kinematics: curved turns, straight cruising, speed variations, multi-agent interactions.
    """
    def __init__(self, num_samples: int = 1200, num_agents: int = 4, seed: int = 42):
        self.num_samples = num_samples
        self.num_agents = num_agents

        self.past_data: List[torch.Tensor] = []
        self.future_data: List[torch.Tensor] = []

        g = torch.Generator()
        g.manual_seed(seed)

        for _ in range(num_samples):
            # Base velocity (8 - 14 m/s approx 30 - 50 km/h) and road curvature
            speed = 8.0 + torch.rand(1, generator=g).item() * 6.0
            dt = 0.1  # 10Hz sampling
            curvature = (torch.rand(1, generator=g).item() - 0.5) * 0.05

            past_traj = []
            cur_x, cur_y, cur_heading = 0.0, 0.0, 0.0

            # Past 20 steps
            for _ in range(20):
                past_traj.append([cur_x, cur_y])
                cur_heading += curvature
                cur_x += speed * dt * math.cos(cur_heading) + (torch.randn(1, generator=g).item() * 0.05)
                cur_y += speed * dt * math.sin(cur_heading) + (torch.randn(1, generator=g).item() * 0.05)

            # Future 30 steps
            future_traj = []
            for _ in range(30):
                cur_heading += curvature
                cur_x += speed * dt * math.cos(cur_heading) + (torch.randn(1, generator=g).item() * 0.08)
                cur_y += speed * dt * math.sin(cur_heading) + (torch.randn(1, generator=g).item() * 0.08)
                future_traj.append([cur_x, cur_y])

            # Multi-agent tensor shape: [T, N, 2]
            past_tensor = torch.tensor(past_traj, dtype=torch.float32).unsqueeze(1).repeat(1, num_agents, 1)
            future_tensor = torch.tensor(future_traj, dtype=torch.float32).unsqueeze(1).repeat(1, num_agents, 1)

            # Add neighbor vehicle lane offsets
            for a in range(1, num_agents):
                offset_x = (a * 4.0) + torch.randn(1, generator=g).item()
                offset_y = (a * 8.0) + torch.randn(1, generator=g).item()
                past_tensor[:, a, 0] += offset_x
                past_tensor[:, a, 1] += offset_y
                future_tensor[:, a, 0] += offset_x
                future_tensor[:, a, 1] += offset_y

            self.past_data.append(past_tensor)
            self.future_data.append(future_tensor)

        # Pre-generate synthetic map lanes [max_lanes=16, lane_pts=10, 2]
        self.lane_pts = 10
        self.max_lanes = 16
        lanes = []
        for l in range(self.max_lanes):
            offset_y = (l - (self.max_lanes // 2)) * 3.5
            xs = torch.linspace(-25.0, 65.0, self.lane_pts)
            ys = torch.ones(self.lane_pts) * offset_y + torch.sin(xs * 0.04) * 1.5
            lane = torch.stack([xs, ys], dim=-1)
            lanes.append(lane)
        self.default_map_lanes = torch.stack(lanes, dim=0)  # [16, 10, 2]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        full_past = self.past_data[idx]    # [20, num_agents, 2]
        full_future = self.future_data[idx] # [30, num_agents, 2]
        return {
            "past": full_past[:, 0, :],              # [20, 2] target agent
            "future": full_future[:, 0, :],          # [30, 2] target agent
            "agents_past": full_past,                # [20, num_agents, 2]
            "map_lanes": self.default_map_lanes      # [16, 10, 2]
        }

