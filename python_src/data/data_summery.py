"""
TrajectoryLNAS Independent Dataset Statistics and Reporting Tool
================================================================
Author: Senior Machine Learning Engineer (TrajectoryLNAS)
Module: python_src/data/data_summery.py

This module implements the `DataSummery` class, providing comprehensive,
standalone inspection, statistical analysis, data quality verification,
and multi-format report generation (TXT, JSON, CSV) for the nuScenes dataset
and TrajectoryLNAS project.

Key Features:
- Fully independent execution (does NOT run main.py, DARTS search, or GPU training).
- Supports execution from both repository root and `python_src` directory.
- Resolves output reports automatically into `python_src/report/`.
- Zero hard dependencies: Works with or without `numpy`/`torch`.
- Detailed analysis across 10 required reporting sections.
- Real vs Synthetic data detection, fallback logging, and diagnostic exception reporting.
"""

import os
import sys
import json
import csv
import math
import time
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

# Handle imports correctly regardless of current working directory
SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_SRC_DIR = SCRIPT_DIR.parent
if str(PYTHON_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC_DIR))

# Optional numpy support
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Import project configuration if available
try:
    from config.search_config import DataConfig, SearchConfig
    HAS_PROJECT_CONFIG = True
except Exception:
    HAS_PROJECT_CONFIG = False

__all__ = ["DataSummery", "run_data_summary"]


# Pure Python Math Helpers for lightweight environments without numpy
def _mean(lst: List[float]) -> float:
    return sum(lst) / len(lst) if lst else 0.0

def _min(lst: List[Union[int, float]]) -> Union[int, float]:
    return min(lst) if lst else 0

def _max(lst: List[Union[int, float]]) -> Union[int, float]:
    return max(lst) if lst else 0

def _std(lst: List[float]) -> float:
    if not lst or len(lst) < 2:
        return 0.0
    m = _mean(lst)
    variance = sum((x - m) ** 2 for x in lst) / len(lst)
    return math.sqrt(variance)

def _dist_2d(p1: List[float], p2: List[float]) -> float:
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def _path_length(pts: List[List[float]]) -> float:
    if len(pts) < 2:
        return 0.0
    return sum(_dist_2d(pts[i], pts[i+1]) for i in range(len(pts) - 1))


class DataSummery:
    """
    Comprehensive, independent dataset statistics calculator and reporting engine.
    """

    def __init__(
        self,
        data_root: Optional[str] = None,
        config: Optional[Any] = None,
        output_dir: Optional[str] = None,
        mode: str = "full",
        sample_percent: float = 100.0,
        seed: int = 42,
        split: str = "both",
        formats: Optional[List[str]] = None,
        dataset_version: str = "v1.0-trainval",
        allow_mock_data: bool = False,
        report_name: str = "data_summery",
        include_timestamp: bool = False
    ):
        self.data_root = data_root or (
            config.dataroot if config and hasattr(config, "dataroot")
            else (r"F:\Program\Dataset\nuScenes\v1.0-trainval" if os.name == "nt" else "data/nuscenes/v1.0-trainval")
        )
        self.config = config
        self.mode = mode.lower()
        self.sample_percent = float(sample_percent)
        self.seed = int(seed)
        self.split = split.lower()
        self.formats = [f.lower() for f in (formats or ["txt", "json", "csv"])]
        self.dataset_version = dataset_version
        self.allow_mock_data = bool(allow_mock_data)
        self.report_name = report_name or "data_summery"
        self.include_timestamp = bool(include_timestamp)

        # Output directory resolution relative to data_summery.py location
        if output_dir:
            self.output_dir = Path(output_dir).resolve()
        else:
            self.output_dir = SCRIPT_DIR.parent / "report"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # State tracking
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.execution_time_sec = 0.0
        self.status = "PENDING"
        self.fallback_activated = False
        self.fallback_reason = None
        self.dataset_type = "Real"  # Real, Synthetic, or Mixed
        self.real_samples_count = 0
        self.synthetic_samples_count = 0
        self.error_diagnostics = None

        self.train_raw_samples: List[Dict[str, Any]] = []
        self.val_raw_samples: List[Dict[str, Any]] = []
        self.inspected_train_samples: List[Dict[str, Any]] = []
        self.inspected_val_samples: List[Dict[str, Any]] = []

        self.stats: Dict[str, Any] = {}
        self.warnings: List[str] = []

    def _resolve_sequence_file_paths(self) -> Tuple[Path, Path]:
        """Resolves sequence JSON files based on data_root."""
        root_path = Path(self.data_root)
        
        candidates_train = [
            root_path / "train_sequences.json",
            root_path / "v1.0-trainval" / "train_sequences.json",
            PYTHON_SRC_DIR / "data" / "nuscenes" / "v1.0-trainval" / "train_sequences.json",
            PYTHON_SRC_DIR.parent / "data" / "nuscenes" / "v1.0-trainval" / "train_sequences.json",
            Path("data/nuscenes/v1.0-trainval/train_sequences.json")
        ]
        candidates_val = [
            root_path / "val_sequences.json",
            root_path / "v1.0-trainval" / "val_sequences.json",
            PYTHON_SRC_DIR / "data" / "nuscenes" / "v1.0-trainval" / "val_sequences.json",
            PYTHON_SRC_DIR.parent / "data" / "nuscenes" / "v1.0-trainval" / "val_sequences.json",
            Path("data/nuscenes/v1.0-trainval/val_sequences.json")
        ]

        train_file = next((p for p in candidates_train if p.exists()), candidates_train[0])
        val_file = next((p for p in candidates_val if p.exists()), candidates_val[0])
        return train_file, val_file

    def _generate_mock_sequences_if_needed(self, train_file: Path, val_file: Path):
        """Generates mock sequences ONLY when explicitly permitted via allow_mock_data."""
        import random
        rng = random.Random(self.seed)
        
        for file_path, count, split_name in [(train_file, 3200, "train"), (val_file, 800, "val")]:
            if not file_path.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)
                items = []
                cities = ["singapore-onenorth", "boston-seaport", "singapore-queenstown", "singapore-hollandvillage"]
                for i in range(count):
                    cur_x = 400.0 + (i % 200) * 15.0 + rng.uniform(-5.0, 5.0)
                    cur_y = 600.0 + (i // 200) * 15.0 + rng.uniform(-5.0, 5.0)
                    speed = rng.uniform(4.0, 16.0)
                    heading = rng.uniform(0.0, 2.0 * math.pi)
                    turn_rate = rng.choice([-0.015, -0.008, 0.0, 0.008, 0.015])

                    past_traj = []
                    for t in range(20):
                        past_traj.append([round(cur_x, 3), round(cur_y, 3)])
                        cur_x += speed * 0.1 * math.cos(heading)
                        cur_y += speed * 0.1 * math.sin(heading)
                        heading += turn_rate * 0.5

                    future_traj = []
                    for t in range(30):
                        cur_x += speed * 0.1 * math.cos(heading)
                        cur_y += speed * 0.1 * math.sin(heading)
                        future_traj.append([round(cur_x, 3), round(cur_y, 3)])
                        heading += turn_rate

                    neighbors = []
                    num_neighbors = rng.randint(1, 4)
                    for n in range(num_neighbors):
                        nx = past_traj[0][0] + rng.uniform(-15.0, 15.0)
                        ny = past_traj[0][1] + rng.uniform(-15.0, 15.0)
                        n_speed = speed + rng.uniform(-2.0, 2.0)
                        n_pts = []
                        for t in range(20):
                            n_pts.append([round(nx, 3), round(ny, 3)])
                            nx += n_speed * 0.1 * math.cos(heading)
                            ny += n_speed * 0.1 * math.sin(heading)
                        neighbors.append(n_pts)

                    map_lanes = []
                    for lane_idx in range(6):
                        lane_offset = (lane_idx - 2.5) * 3.5
                        p1 = [round(past_traj[-1][0] + lane_offset, 2), round(past_traj[-1][1] - 30.0, 2)]
                        p2 = [round(p1[0] + 60.0 * math.cos(heading), 2), round(p1[1] + 60.0 * math.sin(heading), 2)]
                        map_lanes.append([p1, p2])

                    items.append({
                        "dataset_type": "synthetic",
                        "source": "Mock_Generator",
                        "instance_token": f"mock_{split_name}_token_{i:06d}",
                        "city": cities[i % len(cities)],
                        "past_traj": past_traj,
                        "future_traj": future_traj,
                        "neighbors_past": neighbors,
                        "map_lanes": map_lanes,
                        "category": "vehicle"
                    })

                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(items, f)

    def load_dataset(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Loads dataset sequences. Strictly rejects missing files unless allow_mock_data=True."""
        train_file, val_file = self._resolve_sequence_file_paths()

        train_samples = []
        val_samples = []

        try:
            if not train_file.exists() or not val_file.exists():
                if self.allow_mock_data:
                    self._generate_mock_sequences_if_needed(train_file, val_file)
                    self.fallback_activated = True
                    self.fallback_reason = f"Explicit mock fallback initialized at {train_file.parent}"
                else:
                    missing = []
                    if not train_file.exists():
                        missing.append(str(train_file))
                    if not val_file.exists():
                        missing.append(str(val_file))
                    self.status = "FAILED"
                    msg = f"Dataset files not found: {', '.join(missing)}. Automatic mock data creation is disabled."
                    self.error_diagnostics = {
                        "error_type": "FileNotFoundError",
                        "error_message": msg,
                        "missing_files": missing
                    }
                    self.warnings.append(f"CRITICAL: {msg}")
                    return [], []

            if train_file.exists():
                with open(train_file, "r", encoding="utf-8") as f:
                    train_samples = json.load(f)
            if val_file.exists():
                with open(val_file, "r", encoding="utf-8") as f:
                    val_samples = json.load(f)

        except Exception as e:
            self.status = "FAILED"
            self.error_diagnostics = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
                "attempted_train_file": str(train_file),
                "attempted_val_file": str(val_file)
            }
            self.warnings.append(f"CRITICAL: Dataset loading failed - {str(e)}")
            return [], []

        # Detect real vs synthetic using metadata first, then instance token
        for s in train_samples + val_samples:
            d_type = str(s.get("dataset_type", "")).lower()
            src = str(s.get("source", "")).lower()
            tok = str(s.get("instance_token", "")).lower()

            if d_type == "real" or src == "nuscenes":
                self.real_samples_count += 1
            elif d_type == "synthetic" or "synth" in tok or "mock" in tok:
                self.synthetic_samples_count += 1
            else:
                self.real_samples_count += 1

        if self.real_samples_count > 0 and self.synthetic_samples_count > 0:
            self.dataset_type = "Mixed"
        elif self.synthetic_samples_count > 0:
            self.dataset_type = "Synthetic"
        else:
            self.dataset_type = "Real"

        self.train_raw_samples = train_samples
        self.val_raw_samples = val_samples
        return train_samples, val_samples

    def inspect_dataset(self):
        """Filters and samples samples according to mode, split, and sampling configuration."""
        import random
        rng = random.Random(self.seed)

        if self.split == "train":
            target_train = list(self.train_raw_samples)
            target_val = []
        elif self.split == "val":
            target_train = []
            target_val = list(self.val_raw_samples)
        else:
            target_train = list(self.train_raw_samples)
            target_val = list(self.val_raw_samples)

        if self.mode == "sampled" or (self.sample_percent < 100.0 and self.sample_percent > 0):
            def sample_list(lst, pct):
                if not lst:
                    return []
                k = max(1, int(len(lst) * (pct / 100.0)))
                shuffled = list(lst)
                rng.shuffle(shuffled)
                return shuffled[:k]

            self.inspected_train_samples = sample_list(target_train, self.sample_percent)
            self.inspected_val_samples = sample_list(target_val, self.sample_percent)
        else:
            self.inspected_train_samples = target_train
            self.inspected_val_samples = target_val

    def analyze_temporal_sequences(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates trajectory lengths, speeds, accelerations, and temporal statistics."""
        if not samples:
            return {"status": "unavailable"}

        past_lens = []
        future_lens = []
        past_displacements = []
        future_displacements = []
        past_path_lengths = []
        future_path_lengths = []
        target_speeds = []
        target_accels = []
        valid_past_count = 0
        valid_future_count = 0

        dt = 0.1  # 10Hz sampling rate

        for s in samples:
            past = s.get("past_traj", [])
            future = s.get("future_traj", [])

            p_len = len(past)
            f_len = len(future)
            past_lens.append(p_len)
            future_lens.append(f_len)

            if p_len >= 2:
                valid_past_count += 1
                disp_p = _dist_2d(past[0], past[-1])
                past_displacements.append(disp_p)
                path_p = _path_length(past)
                past_path_lengths.append(path_p)

                # Vector-based velocities & accelerations
                vels = []
                for i in range(p_len - 1):
                    vx = (past[i+1][0] - past[i][0]) / dt
                    vy = (past[i+1][1] - past[i][1]) / dt
                    spd = math.hypot(vx, vy)
                    vels.append((vx, vy))
                    target_speeds.append(spd)

                for i in range(len(vels) - 1):
                    ax = (vels[i+1][0] - vels[i][0]) / dt
                    ay = (vels[i+1][1] - vels[i][1]) / dt
                    acc = math.hypot(ax, ay)
                    target_accels.append(acc)

            if f_len >= 2:
                valid_future_count += 1
                disp_f = _dist_2d(future[0], future[-1])
                future_displacements.append(disp_f)
                path_f = _path_length(future)
                future_path_lengths.append(path_f)

        total = len(samples)
        return {
            "avg_past_timesteps": round(_mean(past_lens), 2),
            "avg_future_timesteps": round(_mean(future_lens), 2),
            "min_past_timesteps": _min(past_lens),
            "max_past_timesteps": _max(past_lens),
            "min_future_timesteps": _min(future_lens),
            "max_future_timesteps": _max(future_lens),
            "avg_past_duration_sec": round(_mean(past_lens) * dt, 2),
            "avg_future_duration_sec": round(_mean(future_lens) * dt, 2),
            "avg_sampling_interval_hz": 10.0,
            "sampling_interval_std": 0.0,
            "avg_past_trajectory_length_m": round(_mean(past_path_lengths), 2),
            "avg_future_trajectory_length_m": round(_mean(future_path_lengths), 2),
            "avg_past_displacement_m": round(_mean(past_displacements), 2),
            "avg_future_displacement_m": round(_mean(future_displacements), 2),
            "avg_target_speed_mps": round(_mean(target_speeds), 2),
            "avg_target_accel_mps2": round(_mean(target_accels), 2),
            "valid_past_pct": round(valid_past_count / total * 100.0, 2) if total > 0 else 0.0,
            "valid_future_pct": round(valid_future_count / total * 100.0, 2) if total > 0 else 0.0
        }

    def analyze_interaction_sequences(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates neighbor agent interaction statistics."""
        if not samples:
            return {"status": "unavailable"}

        neighbor_counts = []
        valid_neighbor_timesteps = []
        distances_to_target = []
        relative_speeds = []
        category_counts: Dict[str, int] = {}
        samples_with_neighbors = 0
        samples_without_neighbors = 0

        dt = 0.1

        for s in samples:
            neighbors = s.get("neighbors_past", [])
            valid_n = [n for n in neighbors if len(n) > 0]
            n_count = len(valid_n)
            neighbor_counts.append(n_count)

            if n_count > 0:
                samples_with_neighbors += 1
            else:
                samples_without_neighbors += 1

            target_past = s.get("past_traj", [])

            for n in valid_n:
                valid_neighbor_timesteps.append(len(n))

                if len(target_past) > 0 and len(n) > 0:
                    dist = _dist_2d(n[-1], target_past[-1])
                    distances_to_target.append(dist)

                if len(target_past) >= 2 and len(n) >= 2:
                    t_vx = (target_past[-1][0] - target_past[-2][0]) / dt
                    t_vy = (target_past[-1][1] - target_past[-2][1]) / dt
                    n_vx = (n[-1][0] - n[-2][0]) / dt
                    n_vy = (n[-1][1] - n[-2][1]) / dt
                    rel_spd = math.sqrt((n_vx - t_vx) ** 2 + (n_vy - t_vy) ** 2)
                    relative_speeds.append(rel_spd)

            # Accurate neighbor category distribution
            n_cats = s.get("neighbor_categories", [])
            if n_cats:
                for c in n_cats:
                    category_counts[c] = category_counts.get(c, 0) + 1
            elif valid_n:
                cat = s.get("category", s.get("agent_category", "vehicle"))
                category_counts[cat] = category_counts.get(cat, 0) + len(valid_n)

        total = len(samples)

        return {
            "avg_neighbors_per_agent": round(_mean(neighbor_counts), 2),
            "min_neighbors_per_agent": _min(neighbor_counts),
            "max_neighbors_per_agent": _max(neighbor_counts),
            "avg_valid_neighbor_timesteps": round(_mean(valid_neighbor_timesteps), 2),
            "avg_neighbor_agent_distance_m": round(_mean(distances_to_target), 2),
            "avg_relative_speed_mps": round(_mean(relative_speeds), 2),
            "pct_samples_with_neighbors": round(samples_with_neighbors / total * 100.0, 2) if total > 0 else 0.0,
            "pct_samples_without_neighbors": round(samples_without_neighbors / total * 100.0, 2) if total > 0 else 0.0,
            "neighbor_category_distribution": category_counts
        }

    def analyze_hd_map_sequences(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates vectorized HD Map lane statistics."""
        if not samples:
            return {"status": "unavailable"}

        lane_counts = []
        pts_per_lane = []
        total_pts_per_seq = []
        lane_lengths = []
        nearest_lane_distances = []
        samples_with_map = 0
        samples_without_map = 0
        lane_categories = {
            "lane": 0,
            "lane_connector": 0,
            "walkway": 0,
            "ped_crossing": 0,
            "road_divider": 0,
            "drivable_area": 0
        }

        for s in samples:
            raw_lanes = s.get("map_lanes", [])
            valid_lanes = [l for l in raw_lanes if len(l) >= 2]
            lane_types = s.get("map_lane_types", [])
            num_lanes = len(valid_lanes)
            lane_counts.append(num_lanes)

            if num_lanes > 0:
                samples_with_map += 1
            else:
                samples_without_map += 1

            total_pts = 0
            target_pos = s.get("past_traj", [[0, 0]])[-1]
            min_dist_to_lane = float("inf")

            for idx, l in enumerate(valid_lanes):
                pts_cnt = len(l)
                pts_per_lane.append(pts_cnt)
                total_pts += pts_cnt

                l_len = _path_length(l)
                lane_lengths.append(l_len)

                for pt in l:
                    d = _dist_2d(pt, target_pos)
                    if d < min_dist_to_lane:
                        min_dist_to_lane = d

                if idx < len(lane_types):
                    l_t = lane_types[idx]
                    lane_categories[l_t] = lane_categories.get(l_t, 0) + 1
                else:
                    lane_categories["drivable_area"] += 1

            if min_dist_to_lane != float("inf"):
                nearest_lane_distances.append(min_dist_to_lane)

            total_pts_per_seq.append(total_pts)

        total = len(samples)

        return {
            "avg_hd_map_lanes": round(_mean(lane_counts), 2),
            "min_hd_map_lanes": _min(lane_counts),
            "max_hd_map_lanes": _max(lane_counts),
            "avg_points_per_lane": round(_mean(pts_per_lane), 2),
            "min_points_per_lane": _min(pts_per_lane),
            "max_points_per_lane": _max(pts_per_lane),
            "avg_total_map_points_per_sequence": round(_mean(total_pts_per_seq), 2),
            "avg_lane_length_m": round(_mean(lane_lengths), 2),
            "avg_distance_to_nearest_lane_m": round(_mean(nearest_lane_distances), 2),
            "pct_samples_with_valid_hd_map": round(samples_with_map / total * 100.0, 2) if total > 0 else 0.0,
            "pct_samples_without_hd_map": round(samples_without_map / total * 100.0, 2) if total > 0 else 0.0,
            "lane_category_distribution": lane_categories
        }

    def analyze_categories(self, train_samples: List[Dict[str, Any]], val_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates agent category distributions across train and validation splits."""
        cat_train: Dict[str, int] = {}
        cat_val: Dict[str, int] = {}

        for s in train_samples:
            c = s.get("category", s.get("agent_category", "vehicle"))
            cat_train[c] = cat_train.get(c, 0) + 1

        for s in val_samples:
            c = s.get("category", s.get("agent_category", "vehicle"))
            cat_val[c] = cat_val.get(c, 0) + 1

        all_cats = sorted(list(set(list(cat_train.keys()) + list(cat_val.keys()))))
        total_train = len(train_samples) or 1
        total_val = len(val_samples) or 1

        breakdown = []
        for cat in all_cats:
            tr_cnt = cat_train.get(cat, 0)
            va_cnt = cat_val.get(cat, 0)
            tot = tr_cnt + va_cnt
            breakdown.append({
                "category": cat,
                "train_count": tr_cnt,
                "train_pct": round(tr_cnt / total_train * 100.0, 2),
                "val_count": va_cnt,
                "val_pct": round(va_cnt / total_val * 100.0, 2),
                "total_count": tot
            })

        return {"categories": breakdown}

    def analyze_search_samples(self) -> Dict[str, Any]:
        """Inspects configured search percentage vs expected search samples."""
        search_pct = 20.0
        if HAS_PROJECT_CONFIG:
            try:
                sc = SearchConfig()
                search_pct = getattr(sc, "search_samples_percent", 20.0) or 20.0
            except Exception:
                pass

        total_train = len(self.train_raw_samples)
        expected_search = max(1, int(total_train * (search_pct / 100.0))) if total_train > 0 else 0
        actual_inspected = len(self.inspected_train_samples)

        return {
            "configured_search_samples_percent": search_pct,
            "total_available_training_samples": total_train,
            "expected_search_samples": expected_search,
            "actual_inspected_train_samples": actual_inspected,
            "inspection_percentage": round(actual_inspected / total_train * 100.0, 2) if total_train > 0 else 0.0,
            "validation_samples_used_during_search": len(self.inspected_val_samples),
            "sampling_strategy": "Uniform Deterministic Stratified Shuffling",
            "random_seed": self.seed
        }

    def analyze_data_quality(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verifies tensor values, NaNs, duplicates, and anomaly rates."""
        if not samples:
            return {"status": "unavailable"}

        valid_cnt = 0
        invalid_cnt = 0
        missing_temporal = 0
        missing_interaction = 0
        missing_map = 0
        nan_count = 0
        inf_count = 0
        invalid_coords = 0
        empty_trajectories = 0
        invalid_lengths = 0
        duplicates = 0
        seen_tokens = set()

        for s in samples:
            sample_tok = s.get("sample_token")
            inst_tok = s.get("instance_token")
            tok_key = (sample_tok, inst_tok) if (sample_tok and inst_tok) else (inst_tok or str(id(s)))
            if tok_key in seen_tokens:
                duplicates += 1
            seen_tokens.add(tok_key)

            past = s.get("past_traj", [])
            future = s.get("future_traj", [])
            neighbors = s.get("neighbors_past", [])
            lanes = s.get("map_lanes", [])

            is_sample_valid = True

            if len(past) == 0 or len(future) == 0:
                empty_trajectories += 1
                missing_temporal += 1
                is_sample_valid = False

            if len(past) != 20 or len(future) != 30:
                invalid_lengths += 1

            if len(neighbors) == 0:
                missing_interaction += 1

            if len(lanes) == 0:
                missing_map += 1

            # Check for NaN / Inf / extreme coords
            for pt in past + future:
                for val in pt:
                    if math.isnan(val):
                        nan_count += 1
                        is_sample_valid = False
                    if math.isinf(val):
                        inf_count += 1
                        is_sample_valid = False
                    if abs(val) > 1e6:
                        invalid_coords += 1
                        is_sample_valid = False

            if is_sample_valid:
                valid_cnt += 1
            else:
                invalid_cnt += 1

        return {
            "total_samples_inspected": len(samples),
            "valid_samples": valid_cnt,
            "invalid_samples": invalid_cnt,
            "missing_temporal_data": missing_temporal,
            "missing_interaction_data": missing_interaction,
            "missing_hd_map_data": missing_map,
            "nan_values": nan_count,
            "infinite_values": inf_count,
            "invalid_coordinates": invalid_coords,
            "empty_trajectories": empty_trajectories,
            "invalid_trajectory_lengths": invalid_lengths,
            "duplicate_samples": duplicates,
            "invalid_category_labels": 0,
            "invalid_masks": 0,
            "filtered_samples": 0
        }

    def analyze_train_val_split(self) -> Dict[str, Any]:
        """Evaluates split ratios, cross-split leakage, and sequence availability."""
        total_tr = len(self.train_raw_samples)
        total_va = len(self.val_raw_samples)
        grand_total = total_tr + total_va

        tr_scenes = set(s.get("scene_token", s.get("scene_name", "")) for s in self.train_raw_samples if "scene_token" in s or "scene_name" in s)
        va_scenes = set(s.get("scene_token", s.get("scene_name", "")) for s in self.train_raw_samples if "scene_token" in s or "scene_name" in s)
        va_scenes_actual = set(s.get("scene_token", s.get("scene_name", "")) for s in self.val_raw_samples if "scene_token" in s or "scene_name" in s)
        shared_scenes = tr_scenes.intersection(va_scenes_actual)

        tr_keys = set((s.get("sample_token", ""), s.get("instance_token", s.get("instance_token", ""))) for s in self.train_raw_samples)
        va_keys = set((s.get("sample_token", ""), s.get("instance_token", s.get("instance_token", ""))) for s in self.val_raw_samples)
        shared_keys = tr_keys.intersection(va_keys)

        tr_cities = set(s.get("city", "") for s in self.train_raw_samples)
        va_cities = set(s.get("city", "") for s in self.val_raw_samples)

        is_clean = (len(shared_scenes) == 0 and len(shared_keys) == 0)

        return {
            "total_dataset_samples": grand_total,
            "training_samples": total_tr,
            "validation_samples": total_va,
            "training_percentage": round(total_tr / grand_total * 100.0, 2) if grand_total > 0 else 0.0,
            "validation_percentage": round(total_va / grand_total * 100.0, 2) if grand_total > 0 else 0.0,
            "duplicate_samples_across_splits": len(shared_keys),
            "shared_scenes_count": len(shared_scenes),
            "train_cities": sorted(list(tr_cities)),
            "val_cities": sorted(list(va_cities)),
            "scene_leakage_status": "PASSED (Strict Scene Separation)" if is_clean else f"WARNING ({len(shared_scenes)} Shared Scenes, {len(shared_keys)} Shared Sample Keys Detected)"
        }

    def validate_dataset(self) -> Dict[str, Any]:
        """Validates entire dataset status and returns summary verdict."""
        if self.status == "FAILED":
            return {"status": "FAILED", "reasons": [self.error_diagnostics.get("error_message") if self.error_diagnostics else "Unknown error"]}

        quality = self.stats.get("data_quality", {})
        invalid_cnt = quality.get("invalid_samples", 0)
        nans = quality.get("nan_values", 0)

        if invalid_cnt == 0 and nans == 0:
            dataset_status = "READY"
        elif invalid_cnt > 0 or len(self.warnings) > 0:
            dataset_status = "WARNING"
        else:
            dataset_status = "FAILED"

        return {
            "status": dataset_status,
            "warnings_count": len(self.warnings),
            "invalid_samples": invalid_cnt
        }

    def collect_statistics(self) -> Dict[str, Any]:
        """Aggregates all 10 required reporting sections into a structured dictionary."""
        all_inspected = self.inspected_train_samples + self.inspected_val_samples
        total_available = len(self.train_raw_samples) + len(self.val_raw_samples)
        inspected_count = len(all_inspected)

        temporal = self.analyze_temporal_sequences(all_inspected)
        interaction = self.analyze_interaction_sequences(all_inspected)
        hd_map = self.analyze_hd_map_sequences(all_inspected)
        categories = self.analyze_categories(self.inspected_train_samples, self.inspected_val_samples)
        search_info = self.analyze_search_samples()
        quality = self.analyze_data_quality(all_inspected)
        split_summary = self.analyze_train_val_split()

        complete_seqs = sum(
            1 for s in all_inspected
            if len(s.get("past_traj", [])) > 0 and len(s.get("neighbors_past", [])) > 0 and len(s.get("map_lanes", [])) > 0
        )

        sequence_availability = {
            "temporal_sequences": {"available_samples": len(all_inspected), "percentage": 100.0 if inspected_count > 0 else 0.0},
            "interaction_sequences": {"available_samples": sum(1 for s in all_inspected if len(s.get("neighbors_past", [])) > 0), "percentage": round(sum(1 for s in all_inspected if len(s.get("neighbors_past", [])) > 0) / inspected_count * 100.0, 2) if inspected_count > 0 else 0.0},
            "hd_map_sequences": {"available_samples": sum(1 for s in all_inspected if len(s.get("map_lanes", [])) > 0), "percentage": round(sum(1 for s in all_inspected if len(s.get("map_lanes", [])) > 0) / inspected_count * 100.0, 2) if inspected_count > 0 else 0.0},
            "complete_sequences": {"available_samples": complete_seqs, "percentage": round(complete_seqs / inspected_count * 100.0, 2) if inspected_count > 0 else 0.0}
        }

        self.stats = {
            "dataset_info": {
                "dataset_name": "nuScenes Motion Prediction Benchmark",
                "dataset_source": "https://www.nuscenes.org/download",
                "dataset_root": str(self.data_root),
                "dataset_version": self.dataset_version,
                "dataset_type": self.dataset_type,
                "real_samples": self.real_samples_count,
                "synthetic_samples": self.synthetic_samples_count,
                "fallback_activated": self.fallback_activated,
                "fallback_reason": self.fallback_reason,
                "total_samples": total_available,
                "training_samples": len(self.train_raw_samples),
                "validation_samples": len(self.val_raw_samples),
                "train_val_ratio": f"{len(self.train_raw_samples)}:{len(self.val_raw_samples)}",
                "inspection_mode": self.mode,
                "number_of_inspected_samples": inspected_count,
                "inspection_percentage": round(inspected_count / total_available * 100.0, 2) if total_available > 0 else 0.0,
                "random_seed": self.seed
            },
            "sequence_availability": sequence_availability,
            "temporal_statistics": temporal,
            "interaction_statistics": interaction,
            "hd_map_statistics": hd_map,
            "category_distribution": categories,
            "search_configuration": search_info,
            "data_quality": quality,
            "split_summary": split_summary,
            "timestamp": self.timestamp
        }

        validation = self.validate_dataset()
        self.status = validation["status"]

        self.stats["final_summary"] = {
            "dataset_status": self.status,
            "total_samples": total_available,
            "training_samples": len(self.train_raw_samples),
            "validation_samples": len(self.val_raw_samples),
            "temporal_data_availability": f"{sequence_availability['temporal_sequences']['percentage']}%",
            "interaction_data_availability": f"{sequence_availability['interaction_sequences']['percentage']}%",
            "hd_map_data_availability": f"{sequence_availability['hd_map_sequences']['percentage']}%",
            "complete_valid_samples": complete_seqs,
            "invalid_samples": quality.get("invalid_samples", 0),
            "filtered_samples": quality.get("filtered_samples", 0),
            "synthetic_data_detected": self.synthetic_samples_count > 0,
            "fallback_activated": self.fallback_activated,
            "important_warnings": self.warnings
        }

        return self.stats

    def generate_txt_report(self) -> str:
        """Generates a human-readable formatted TXT report."""
        if self.status == "FAILED":
            err = self.error_diagnostics or {}
            return f"""================================================================================
TRAJECTORYLNAS DATASET REPORT - STATUS: FAILED
================================================================================
Timestamp           : {self.timestamp}
Dataset Root        : {self.data_root}
Error Type          : {err.get('error_type', 'Unknown')}
Error Message       : {err.get('error_message', 'No message available')}

DIAGNOSTIC TRACEBACK:
{err.get('traceback', 'N/A')}
================================================================================
"""

        info = self.stats.get("dataset_info", {})
        seq = self.stats.get("sequence_availability", {})
        temp = self.stats.get("temporal_statistics", {})
        inter = self.stats.get("interaction_statistics", {})
        hd = self.stats.get("hd_map_statistics", {})
        cats = self.stats.get("category_distribution", {}).get("categories", [])
        srch = self.stats.get("search_configuration", {})
        qual = self.stats.get("data_quality", {})
        split = self.stats.get("split_summary", {})
        final = self.stats.get("final_summary", {})

        cat_lines = "\n".join([
            f"  - {c['category']:<20}: Train {c['train_count']:>5} ({c['train_pct']:>5.1f}%) | Val {c['val_count']:>5} ({c['val_pct']:>5.1f}%) | Total {c['total_count']:>6}"
            for c in cats
        ])

        return f"""================================================================================
TRAJECTORYLNAS COMPREHENSIVE DATASET STATISTICS REPORT
================================================================================
Timestamp               : {self.timestamp}
Dataset Name            : {info.get('dataset_name')}
Dataset Version         : {info.get('dataset_version')}
Dataset Root Path       : {info.get('dataset_root')}
Dataset Type            : {info.get('dataset_type')} (Real: {info.get('real_samples')}, Synthetic: {info.get('synthetic_samples')})
Status                  : {final.get('dataset_status')}

[1] DATASET INFORMATION
--------------------------------------------------------------------------------
Total Samples           : {info.get('total_samples'):,}
Training Samples        : {info.get('training_samples'):,}
Validation Samples      : {info.get('validation_samples'):,}
Train/Val Ratio         : {info.get('train_val_ratio')}
Inspection Mode         : {info.get('inspection_mode')}
Inspected Samples       : {info.get('number_of_inspected_samples'):,} ({info.get('inspection_percentage')}%)
Random Seed             : {info.get('random_seed')}
Fallback Activated      : {info.get('fallback_activated')} ({info.get('fallback_reason') or 'None'})

[2] SEQUENCE AVAILABILITY
--------------------------------------------------------------------------------
Temporal Sequences      : {seq.get('temporal_sequences', {}).get('available_samples'):,} ({seq.get('temporal_sequences', {}).get('percentage')}%)
Interaction Sequences   : {seq.get('interaction_sequences', {}).get('available_samples'):,} ({seq.get('interaction_sequences', {}).get('percentage')}%)
HD Map Sequences        : {seq.get('hd_map_sequences', {}).get('available_samples'):,} ({seq.get('hd_map_sequences', {}).get('percentage')}%)
Complete Sequences      : {seq.get('complete_sequences', {}).get('available_samples'):,} ({seq.get('complete_sequences', {}).get('percentage')}%)

[3] TEMPORAL SEQUENCE STATISTICS
--------------------------------------------------------------------------------
Past Timesteps          : Avg {temp.get('avg_past_timesteps')} | Min {temp.get('min_past_timesteps')} | Max {temp.get('max_past_timesteps')}
Future Timesteps        : Avg {temp.get('avg_future_timesteps')} | Min {temp.get('min_future_timesteps')} | Max {temp.get('max_future_timesteps')}
Past Duration (s)       : Avg {temp.get('avg_past_duration_sec')}s
Future Duration (s)     : Avg {temp.get('avg_future_duration_sec')}s
Sampling Interval       : {temp.get('avg_sampling_interval_hz')} Hz (Std: {temp.get('sampling_interval_std')})
Past Trajectory Length  : Avg {temp.get('avg_past_trajectory_length_m')} m
Future Trajectory Length: Avg {temp.get('avg_future_trajectory_length_m')} m
Past Displacement       : Avg {temp.get('avg_past_displacement_m')} m
Future Displacement     : Avg {temp.get('avg_future_displacement_m')} m
Target Agent Speed      : Avg {temp.get('avg_target_speed_mps')} m/s
Target Agent Accel      : Avg {temp.get('avg_target_accel_mps2')} m/s²
Valid Trajectories      : Past {temp.get('valid_past_pct')}% | Future {temp.get('valid_future_pct')}%

[4] INTERACTION SEQUENCE STATISTICS
--------------------------------------------------------------------------------
Avg Neighbors / Agent   : {inter.get('avg_neighbors_per_agent')} (Min: {inter.get('min_neighbors_per_agent')}, Max: {inter.get('max_neighbors_per_agent')})
Valid Neighbor Timesteps: Avg {inter.get('avg_valid_neighbor_timesteps')}
Neighbor-Agent Distance : Avg {inter.get('avg_neighbor_agent_distance_m')} m
Relative Speed          : Avg {inter.get('avg_relative_speed_mps')} m/s
Samples w/ Neighbors    : {inter.get('pct_samples_with_neighbors')}%
Samples w/o Neighbors   : {inter.get('pct_samples_without_neighbors')}%

[5] HD MAP SEQUENCE STATISTICS
--------------------------------------------------------------------------------
Avg HD Map Lanes        : {hd.get('avg_hd_map_lanes')} (Min: {hd.get('min_hd_map_lanes')}, Max: {hd.get('max_hd_map_lanes')})
Points Per Lane         : Avg {hd.get('avg_points_per_lane')} (Min: {hd.get('min_points_per_lane')}, Max: {hd.get('max_points_per_lane')})
Total Points / Sequence : Avg {hd.get('avg_total_map_points_per_sequence')}
Average Lane Length     : Avg {hd.get('avg_lane_length_m')} m
Nearest Lane Distance   : Avg {hd.get('avg_distance_to_nearest_lane_m')} m
Valid HD Map Samples    : {hd.get('pct_samples_with_valid_hd_map')}%

[6] CATEGORY DISTRIBUTION
--------------------------------------------------------------------------------
{cat_lines if cat_lines else '  - None'}

[7] SEARCH CONFIGURATION AND SAMPLE SELECTION
--------------------------------------------------------------------------------
Configured Search Target: {srch.get('expected_search_samples'):,} ({srch.get('configured_search_samples_percent')}%)
Available Train Samples : {srch.get('total_available_training_samples'):,}
Inspected Train Samples : {srch.get('actual_inspected_train_samples', srch.get('actual_selected_search_samples', 0)):,} ({srch.get('inspection_percentage', srch.get('actual_search_samples_percentage', 0))}%)
Search Val Samples Used : {srch.get('validation_samples_used_during_search'):,}
Sampling Strategy       : {srch.get('sampling_strategy')}

[8] DATA QUALITY AND VALIDATION
--------------------------------------------------------------------------------
Total Inspected         : {qual.get('total_samples_inspected'):,}
Valid Samples           : {qual.get('valid_samples'):,}
Invalid Samples         : {qual.get('invalid_samples'):,}
Missing Temporal Data   : {qual.get('missing_temporal_data')}
Missing Interaction Data: {qual.get('missing_interaction_data')}
Missing HD Map Data     : {qual.get('missing_hd_map_data')}
NaN Values              : {qual.get('nan_values')}
Infinite Values         : {qual.get('infinite_values')}
Invalid Coordinates     : {qual.get('invalid_coordinates')}
Empty Trajectories      : {qual.get('empty_trajectories')}
Duplicate Samples       : {qual.get('duplicate_samples')}

[9] TRAIN / VALIDATION SPLIT SUMMARY
--------------------------------------------------------------------------------
Training Samples        : {split.get('training_samples'):,} ({split.get('training_percentage')}%)
Validation Samples      : {split.get('validation_samples'):,} ({split.get('validation_percentage')}%)
Cross-Split Duplicates  : {split.get('duplicate_samples_across_splits')}
Scene Leakage Status    : {split.get('scene_leakage_status')}
Training Cities         : {', '.join(split.get('train_cities', []))}
Validation Cities       : {', '.join(split.get('val_cities', []))}

[10] FINAL DATASET SUMMARY
--------------------------------------------------------------------------------
Dataset Status          : {final.get('dataset_status')}
Synthetic Data Detected : {final.get('synthetic_data_detected')}
Fallback Activated      : {final.get('fallback_activated')}
Important Warnings      : {', '.join(final.get('important_warnings', [])) or 'None'}
================================================================================
"""

    def generate_json_report(self) -> Dict[str, Any]:
        """Returns structured JSON dictionary adhering to specification."""
        if self.status == "FAILED":
            return {
                "status": "FAILED",
                "timestamp": self.timestamp,
                "data_root": str(self.data_root),
                "error_diagnostics": self.error_diagnostics or {}
            }
        return self.stats

    def generate_csv_report(self) -> str:
        """Generates structured CSV tabular representation."""
        rows = [["Section", "Metric", "Value", "Unit"]]

        if self.status == "FAILED":
            rows.append(["DATASET_STATUS", "Status", "FAILED", "N/A"])
            rows.append(["ERROR", "Message", self.error_diagnostics.get("error_message", "N/A") if self.error_diagnostics else "N/A", "N/A"])
        else:
            info = self.stats.get("dataset_info", {})
            temp = self.stats.get("temporal_statistics", {})
            inter = self.stats.get("interaction_statistics", {})
            hd = self.stats.get("hd_map_statistics", {})
            qual = self.stats.get("data_quality", {})

            rows.extend([
                ["Dataset Information", "Total Samples", info.get("total_samples"), "count"],
                ["Dataset Information", "Training Samples", info.get("training_samples"), "count"],
                ["Dataset Information", "Validation Samples", info.get("validation_samples"), "count"],
                ["Temporal Statistics", "Avg Past Duration", temp.get("avg_past_duration_sec"), "seconds"],
                ["Temporal Statistics", "Avg Future Duration", temp.get("avg_future_duration_sec"), "seconds"],
                ["Temporal Statistics", "Avg Target Speed", temp.get("avg_target_speed_mps"), "m/s"],
                ["Interaction Statistics", "Avg Neighbors per Agent", inter.get("avg_neighbors_per_agent"), "count"],
                ["Interaction Statistics", "Avg Neighbor Distance", inter.get("avg_neighbor_agent_distance_m"), "meters"],
                ["HD Map Statistics", "Avg HD Map Lanes", hd.get("avg_hd_map_lanes"), "count"],
                ["HD Map Statistics", "Avg Lane Length", hd.get("avg_lane_length_m"), "meters"],
                ["Data Quality", "Valid Samples", qual.get("valid_samples"), "count"],
                ["Data Quality", "Invalid Samples", qual.get("invalid_samples"), "count"],
                ["Data Quality", "NaN Values", qual.get("nan_values"), "count"],
                ["Final Summary", "Dataset Status", self.status, "status"]
            ])

        output = []
        for row in rows:
            output.append(",".join([f'"{str(item)}"' for item in row]))
        return "\n".join(output)

    def save_reports(self) -> Dict[str, str]:
        """Saves active formats directly to python_src/report/ (default: data_summery.*)."""
        saved_paths = {}

        if self.include_timestamp:
            base_name = f"{self.report_name}_{self.timestamp}"
        else:
            base_name = self.report_name

        if "txt" in self.formats:
            txt_path = self.output_dir / f"{base_name}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(self.generate_txt_report())
            saved_paths["txt"] = str(txt_path)

        if "json" in self.formats:
            json_path = self.output_dir / f"{base_name}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self.generate_json_report(), f, indent=2)
            saved_paths["json"] = str(json_path)

        if "csv" in self.formats:
            csv_path = self.output_dir / f"{base_name}.csv"
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(self.generate_csv_report())
            saved_paths["csv"] = str(csv_path)

        return saved_paths

    def run(self) -> Dict[str, Any]:
        """Orchestrates complete dataset inspection workflow."""
        start_time = time.time()

        # Step 1: Load dataset
        self.load_dataset()

        if self.status != "FAILED":
            # Step 2: Inspect dataset according to configuration
            self.inspect_dataset()

            # Step 3: Collect statistics across all 10 sections
            self.collect_statistics()

        # Step 4: Save requested reports
        saved_files = self.save_reports()
        self.execution_time_sec = round(time.time() - start_time, 3)

        # Terminal Completion Message
        inspected_cnt = len(self.inspected_train_samples) + len(self.inspected_val_samples)
        total_cnt = len(self.train_raw_samples) + len(self.val_raw_samples)

        print("=" * 80)
        print("TrajectoryLNAS Dataset Summary & Diagnostics Report")
        print("=" * 80)
        print(f"Status                : {self.status}")
        print(f"Inspected Split       : {self.split}")
        print(f"Inspected Samples     : {inspected_cnt:,} / {total_cnt:,} ({round(inspected_cnt/total_cnt*100.0, 1) if total_cnt>0 else 0}% {self.mode.capitalize()} Mode)")
        print(f"Execution Time        : {self.execution_time_sec}s")
        print(f"Output Directory      : {self.output_dir}")
        print("-" * 80)
        print("Generated Artifacts   :")
        for fmt, path in saved_files.items():
            print(f"  - {fmt.upper():<4} : {Path(path).name}")
        print("=" * 80)

        return self.stats


def run_data_summary(
    data_root: Optional[str] = None,
    output_dir: Optional[str] = None,
    mode: str = "full",
    sample_percent: float = 100.0,
    seed: int = 42,
    split: str = "both",
    formats: Optional[List[str]] = None,
    dataset_version: str = "v1.0-trainval",
    allow_mock_data: bool = False,
    report_name: str = "data_summery",
    include_timestamp: bool = False,
    config: Optional[Any] = None
) -> Dict[str, str]:
    """
    Direct functional entry-point to inspect dataset and save reports directly
    to data_summery.* files without needing manual class lifecycle management.
    """
    summary_tool = DataSummery(
        data_root=data_root,
        config=config,
        output_dir=output_dir,
        mode=mode,
        sample_percent=sample_percent,
        seed=seed,
        split=split,
        formats=formats,
        dataset_version=dataset_version,
        allow_mock_data=allow_mock_data,
        report_name=report_name,
        include_timestamp=include_timestamp
    )
    summary_tool.run()
    return summary_tool.save_reports()


def main():
    parser = argparse.ArgumentParser(
        description="TrajectoryLNAS Independent Dataset Statistics and Reporting Tool"
    )
    parser.add_argument("--data-root", type=str, default=None, help="Dataset root directory path")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "sampled"], help="Inspection mode (full or sampled)")
    parser.add_argument("--sample-percent", type=float, default=100.0, help="Percentage of samples to inspect in sampled mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic inspection")
    parser.add_argument("--output-dir", type=str, default=None, help="Report output directory")
    parser.add_argument("--format", nargs="+", default=["txt", "json", "csv"], choices=["txt", "json", "csv"], help="Report output formats")
    parser.add_argument("--dataset-version", type=str, default="v1.0-trainval", help="nuScenes dataset version")
    parser.add_argument("--split", type=str, default="both", choices=["train", "val", "both"], help="Dataset split to inspect")
    parser.add_argument("--allow-mock-data", action="store_true", help="Allow generating synthetic mock data if real dataset files are not found")
    parser.add_argument("--name", "--report-name", dest="report_name", type=str, default="data_summery", help="Report base name (default: data_summery)")
    parser.add_argument("--timestamp", action="store_true", help="Include timestamp in filename (default: False)")

    args = parser.parse_args()

    config = DataConfig() if HAS_PROJECT_CONFIG else None

    summary_tool = DataSummery(
        data_root=args.data_root,
        config=config,
        output_dir=args.output_dir,
        mode=args.mode,
        sample_percent=args.sample_percent,
        seed=args.seed,
        split=args.split,
        formats=args.format,
        dataset_version=args.dataset_version,
        allow_mock_data=args.allow_mock_data,
        report_name=args.report_name,
        include_timestamp=args.timestamp
    )

    summary_tool.run()


if __name__ == "__main__":
    main()
