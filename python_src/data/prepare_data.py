"""
TrajectoryLNAS Production Data Preparation Utility (SOLID Architecture)
======================================================================
Processes raw nuScenes v1.0-trainval benchmark files into 10Hz trajectory sequences
(20 timesteps past, 30 timesteps future, surrounding neighbors, HD map lanes).

Follows SOLID Principles:
  - Single Responsibility (SRP): Separate classes for Map validation, Real extraction,
    Synthetic generation, and JSON Dataset writing.
  - Open/Closed (OCP): BaseTrajectoryExtractor abstract interface allows adding Argoverse/Waymo.
  - Liskov Substitution (LSP): Real and Synthetic extractors are interchangeable.
  - Interface Segregation (ISP): Dedicated, focused interfaces.
  - Dependency Inversion (DIP): High-level DataPreparationPipeline depends on abstractions.

Author: Alireza Ghafouri (Student ID: 4031631209)
Advisors: Dr. Zeinab Ghasemi Naraaghi & Dr. Zeinab Abedini
"""

import os
import sys
import json
import math
import random
import argparse
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np

# Single Source of Truth configuration import
try:
    from config.search_config import DataConfig
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from config.search_config import DataConfig


# ==============================================================================
# Helper Utilities
# ==============================================================================

def interpolate_traj(traj: List[List[float]], target_len: int) -> List[List[float]]:
    """Interpolates trajectory coordinates to standardized 10Hz target length."""
    if len(traj) == 0:
        return traj
    arr = np.array(traj, dtype=np.float32)
    x = np.linspace(0, len(arr) - 1, num=len(arr))
    x_new = np.linspace(0, len(arr) - 1, num=target_len)
    interpolated = np.zeros((target_len, 2), dtype=np.float32)
    interpolated[:, 0] = np.interp(x_new, x, arr[:, 0])
    interpolated[:, 1] = np.interp(x_new, x, arr[:, 1])
    return interpolated.tolist()


def matches_category(cat_name: str, target_categories: Optional[Union[str, List[str]]]) -> bool:
    """
    Checks whether a nuScenes annotation category_name matches the requested filter.
    Supports single category string, lists, exact names, or prefixes/substrings
    (e.g., 'vehicle', 'pedestrian', 'human.pedestrian.adult', 'vehicle.car').
    """
    if target_categories is None:
        return True
    if isinstance(target_categories, str):
        target_categories = [target_categories]
    cleaned = [c.strip().lower() for c in target_categories if c and c.strip()]
    if not cleaned:
        return True
    cat_lower = (cat_name or "").lower()
    return any(target in cat_lower for target in cleaned)


def extract_map_lanes_real(nusc_map: Any, x: float, y: float, radius: float = 50.0, max_lanes: int = 20) -> List[List[List[float]]]:
    """Safely extracts real vectorized HD map lane dividers within search radius."""
    lanes = []
    try:
        records = nusc_map.get_records_in_radius(x, y, radius, ["lane_divider", "road_divider"])
        for layer in ["lane_divider", "road_divider"]:
            for token in records.get(layer, []):
                record = nusc_map.get(layer, token)
                node1 = nusc_map.get("node", record["node1_token"])
                node2 = nusc_map.get("node", record["node2_token"])
                lanes.append([[node1["x"], node1["y"]], [node2["x"], node2["y"]]])
                if len(lanes) >= max_lanes:
                    break
            if len(lanes) >= max_lanes:
                break
    except Exception:
        pass
    return lanes[:max_lanes]


def print_dataset_summary(
    train_items: List[Dict[str, Any]],
    val_items: List[Dict[str, Any]],
    train_path: Optional[str] = None,
    val_path: Optional[str] = None,
    category_filter: Optional[Union[str, List[str]]] = None
):
    """Prints a structured academic summary of extracted sequences and dataset attributes."""
    total = len(train_items) + len(val_items)
    all_items = train_items + val_items
    cities: Dict[str, int] = {}
    categories: Dict[str, int] = {}
    tot_neighbors = 0
    tot_lanes = 0

    for it in all_items:
        c = it.get("city", "unknown")
        cities[c] = cities.get(c, 0) + 1
        cat = it.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
        tot_neighbors += len(it.get("neighbors_past", []))
        tot_lanes += len(it.get("map_lanes", []))

    cat_filter_str = str(category_filter) if category_filter is not None else "All Categories"
    num_cats = len(categories)
    if category_filter is None:
        cat_filter_str = f"All Categories ({num_cats})"

    print("=" * 80)
    print("[Data Preparation Summary] Sequence Extraction Completed Successfully")
    print(f" Total Train Sequences: {len(train_items):,}")
    print(f" Total Val Sequences:  {len(val_items):,}")
    print(f" Total Combined Sequences:  {total:,}")
    print(f" Applied Category Filter:   {cat_filter_str}")
    if total > 0:
        print(f" Average Neighbors/Agent:   {tot_neighbors / total:.2f} surrounding agents")
        print(f" Average HD Map Lanes: {tot_lanes / total:.2f} vector lanes / agent")
    print("-" * 80)
    print(" City Distribution:")
    for city, cnt in sorted(cities.items(), key=lambda x: -x[1]):
        pct = (cnt / total * 100.0) if total > 0 else 0
        print(f"   - {city:<22} :   {cnt:5,d} ({pct:5.1f}%)")
    print("-" * 80)
    print(" Agent Category Breakdown:")
    for cat, cnt in sorted(categories.items(), key=lambda x: -x[1]):
        pct = (cnt / total * 100.0) if total > 0 else 0
        print(f"   - {cat:<35}: {cnt:5,d} ({pct:5.1f}%)")
    print("=" * 80)




# ==============================================================================
# Interfaces & Abstractions (ISP & OCP)
# ==============================================================================

class IMapExpansionValidator(ABC):
    """Interface responsible exclusively for validating HD map expansion assets (SRP)."""
    @abstractmethod
    def is_map_expansion_available(self, dataroot: str) -> Tuple[bool, Optional[str], str]:
        """Returns (is_available, map_folder_path, diagnostic_message)."""
        pass


class BaseTrajectoryExtractor(ABC):
    """Abstract Base Class for trajectory data extraction (OCP & LSP)."""
    @abstractmethod
    def extract_sequences(
        self,
        dataroot: str,
        target_count: Optional[int] = 6400,
        seed: int = 42,
        category: Optional[Union[str, List[str]]] = None,
        max_train_scenes: Optional[int] = None,
        max_val_scenes: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extracts and returns (train_sequences, val_sequences).
        If target_count is None, extracts ALL available samples across all scenes without limitation.
        """
        pass


class IDatasetWriter(ABC):
    """Interface for writing processed sequence collections to disk (SRP)."""
    @abstractmethod
    def write(self, train_items: List[Dict[str, Any]], val_items: List[Dict[str, Any]], output_dir: str, seed: int = 42) -> Tuple[str, str]:
        """Saves shuffled dataset files and returns file paths."""
        pass


# ==============================================================================
# Concrete Implementations (SRP & LSP)
# ==============================================================================

class NuScenesMapExpansionValidator(IMapExpansionValidator):
    """
    Validates whether the official nuScenes HD Map expansion assets (v1.3)
    are present on disk in maps/expansion or maps/.
    """
    EXPECTED_CITIES = [
        "boston-seaport",
        "singapore-onenorth",
        "singapore-queenstown",
        "singapore-hollandvillage"
    ]

    def is_map_expansion_available(self, dataroot: str) -> Tuple[bool, Optional[str], str]:
        if not os.path.exists(dataroot):
            return False, None, f"Dataroot not found: {dataroot}"

        # Candidate paths where map expansion JSON files reside
        candidates = [
            os.path.join(dataroot, "maps", "expansion"),
            os.path.join(dataroot, "maps"),
            os.path.join(dataroot, "map_expansion"),
        ]

        for cand in candidates:
            if os.path.exists(cand) and os.path.isdir(cand):
                found_cities = [
                    city for city in self.EXPECTED_CITIES
                    if os.path.exists(os.path.join(cand, f"{city}.json"))
                ]
                if len(found_cities) >= 2:
                    return True, cand, f"Found {len(found_cities)} HD map expansion files in: {cand}"

        return False, None, "No valid maps/expansion JSON layers found in dataroot."


class NuScenesRealDataExtractor(BaseTrajectoryExtractor):
    """
    Extracts 100% real agent trajectories using official nuscenes-devkit (PredictHelper & NuScenesMap).
    Subsamples and uniformly shuffles target_count sequences across all scenes.
    Fully supports agent category filtering (e.g., pedestrian, vehicle) and detailed progress logging.
    """
    def __init__(
        self,
        config: Optional[DataConfig] = None,
        max_train_scenes: Optional[int] = None,
        max_val_scenes: Optional[int] = None,
        category: Optional[Union[str, List[str]]] = None
    ):
        self.config = config or DataConfig()
        self.max_train_scenes = max_train_scenes if max_train_scenes is not None else self.config.max_train_scenes
        self.max_val_scenes = max_val_scenes if max_val_scenes is not None else self.config.max_val_scenes
        self.category = category if category is not None else self.config.category

    def extract_sequences(
        self,
        dataroot: str,
        target_count: Optional[int] = 6400,
        seed: int = 42,
        category: Optional[Union[str, List[str]]] = None,
        max_train_scenes: Optional[int] = None,
        max_val_scenes: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        try:
            from nuscenes.nuscenes import NuScenes
            from nuscenes.prediction import PredictHelper
            from nuscenes.map_expansion.map_api import NuScenesMap
        except ImportError as e:
            raise ImportError(f"nuscenes-devkit is required for real extraction: {e}")

        active_category = category if category is not None else self.category
        active_train_scenes_limit = max_train_scenes if max_train_scenes is not None else self.max_train_scenes
        active_val_scenes_limit = max_val_scenes if max_val_scenes is not None else self.max_val_scenes

        count_desc = f"{target_count:,} shuffled sequences" if target_count is not None else "ALL available sequences (No truncation)"
        cat_desc = str(active_category) if active_category is not None else "All Agent Categories"

        print("=" * 80)
        print(" [Real Data Extractor] Extracting sequences via official nuScenes SDK...")
        print(f" Dataroot: {dataroot}")
        print(f" Target Count: {count_desc}")
        print(f" Category Filter:  {cat_desc}")
        print("=" * 80)

        nusc = NuScenes(version="v1.0-trainval", dataroot=dataroot, verbose=False)
        helper = PredictHelper(nusc)

        cat_names_str = ""
        if hasattr(nusc, "category"):
            cat_list = [c["name"] for c in nusc.category]
            cat_names_str = f"\n  -> Total Category: {len(nusc.category)}, '{', '.join(cat_list)}'"

        print(f"[NuScenes SDK] Metadata Inspection for '{dataroot}':")
        print(f"  -> Total Scenes: {len(nusc.scene)}")
        print(f"  -> Total Keyframe Samples: {len(nusc.sample)}")
        print(f"  -> Total Annotations: {len(nusc.sample_annotation)}")
        print(f"  -> Total Instances:   {len(nusc.instance)}")
        print(f"  -> Total Log Tokens:  {len(nusc.log)}" + cat_names_str)
        print("=" * 80)

        # Load HD Maps
        map_cities = ["boston-seaport", "singapore-onenorth", "singapore-queenstown", "singapore-hollandvillage"]
        nusc_maps = {}
        for city in map_cities:
            try:
                nusc_maps[city] = NuScenesMap(dataroot=dataroot, map_name=city)
            except Exception as e:
                pass
        print(f"[NuScenes Maps] Loaded {len(nusc_maps)} vector HD map layers: {list(nusc_maps.keys())}\n")


        def extract_split(split_name: str, scenes: list, desired_count: Optional[int]) -> List[Dict[str, Any]]:
            items = []
            target_desc = f"{desired_count:,}" if desired_count is not None else "ALL available"
            print(f"[{split_name} Split] Commencing extraction across {len(scenes)} scenes (Target: {target_desc})...")

            for scene_idx, scene in enumerate(scenes):
                sample_token = scene["first_sample_token"]
                log = nusc.get("log", scene["log_token"])
                location = log["location"]
                nusc_map = nusc_maps.get(location)
                scene_seq_count = 0
                scene_name = scene.get("name", f"scene_{scene_idx}")

                while sample_token:
                    sample = nusc.get("sample", sample_token)
                    for ann_token in sample["anns"]:
                        ann = nusc.get("sample_annotation", ann_token)
                        cat_name = ann.get("category_name", "")

                        # Filter by category if requested
                        if not matches_category(cat_name, active_category):
                            continue

                        instance_token = ann["instance_token"]

                        past = helper.get_past_for_agent(instance_token, sample_token, seconds=2, in_agent_frame=False)
                        future = helper.get_future_for_agent(instance_token, sample_token, seconds=3, in_agent_frame=False)

                        if len(past) >= 4 and len(future) >= 6:
                            past_10hz = interpolate_traj(past, 20)
                            future_10hz = interpolate_traj(future, 30)

                            # Neighbors
                            neighbors = []
                            for neighbor_ann_token in sample["anns"]:
                                if neighbor_ann_token == ann_token:
                                    continue
                                neighbor_ann = nusc.get("sample_annotation", neighbor_ann_token)
                                neighbor_instance = neighbor_ann["instance_token"]
                                neighbor_past = helper.get_past_for_agent(neighbor_instance, sample_token, seconds=2, in_agent_frame=False)
                                if len(neighbor_past) >= 4:
                                    neighbors.append(interpolate_traj(neighbor_past, 20))
                                    if len(neighbors) >= 5:
                                        break

                            # Real HD Map polylines
                            x_curr, y_curr = ann["translation"][0], ann["translation"][1]
                            map_lanes = []
                            if nusc_map:
                                map_lanes = extract_map_lanes_real(nusc_map, x_curr, y_curr, radius=50.0, max_lanes=12)

                            items.append({
                                "instance_token": instance_token,
                                "category": cat_name,
                                "city": location,
                                "scene_name": scene_name,
                                "past_traj": past_10hz,
                                "future_traj": future_10hz,
                                "neighbors_past": neighbors,
                                "map_lanes": map_lanes
                            })
                            scene_seq_count += 1

                    sample_token = sample["next"]

                # Per-scene detailed logging
                print(f"  [{split_name} Scene {scene_idx + 1:03d}/{len(scenes):03d} | {scene_name:<16} | {location:<24}]: "
                      f"Extracted {scene_seq_count:3d} valid sequences | Cumulative: {len(items):,}")

            # Deterministic uniform shuffle across all scenes
            rng = random.Random(seed)
            rng.shuffle(items)
            if desired_count is not None and len(items) > desired_count:
                print(f"[{split_name} Split] Subsampling {desired_count:,} sequences from {len(items):,} extracted (uniform shuffle seed={seed}).")
                items = items[:desired_count]
            else:
                print(f"[{split_name} Split] Retained all {len(items):,} extracted sequences.")
            return items

        train_scenes = nusc.scene[:active_train_scenes_limit] if active_train_scenes_limit else nusc.scene
        val_scenes = (
            nusc.scene[700:700 + active_val_scenes_limit]
            if (len(nusc.scene) >= 850 and active_val_scenes_limit)
            else (nusc.scene[700:] if len(nusc.scene) >= 850 else nusc.scene[len(train_scenes):])
        )

        val_target = (max(600, target_count // 4) if target_count is not None else None)
        train_items = extract_split("Train", train_scenes, desired_count=target_count)
        val_items = extract_split("Val", val_scenes, desired_count=val_target)

        return train_items, val_items


class SyntheticTrajectoryGenerator(BaseTrajectoryExtractor):
    """
    Zero-dependency kinematics simulator for development fallback when maps/expansion is absent.
    Generates target_count diverse, kinematically sound trajectories with simulated lane polylines.
    """
    def extract_sequences(
        self,
        dataroot: str,
        target_count: Optional[int] = 6400,
        seed: int = 42,
        category: Optional[Union[str, List[str]]] = None,
        max_train_scenes: Optional[int] = None,
        max_val_scenes: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        count = target_count if target_count is not None else 6400
        count_desc = f"{count:,} shuffled sequences" + (" (Fallback default ceiling)" if target_count is None else "")
        cat_desc = str(category) if category is not None else "All Agent Categories"

        print("=" * 80)
        print(" [Synthetic Generator] Synthesizing diverse kinematics trajectories (Fallback Mode)...")
        print(f" Target Count:     {count_desc}")
        print(f" Category Filter:  {cat_desc}")
        print("=" * 80)

        rng = random.Random(seed)
        cities = ["boston-seaport", "singapore-onenorth", "singapore-queenstown", "singapore-hollandvillage"]

        is_ped = False
        if category is not None:
            cat_str = str(category).lower()
            if "ped" in cat_str or "human" in cat_str:
                is_ped = True

        def generate_collection(count: int, split: str) -> List[Dict[str, Any]]:
            items = []
            for i in range(count):
                cur_x = 400.0 + (i % 200) * 15.0 + rng.uniform(-5.0, 5.0)
                cur_y = 600.0 + (i // 200) * 15.0 + rng.uniform(-5.0, 5.0)

                if is_ped:
                    speed = rng.uniform(0.8, 2.0)  # Walking speed (m/s)
                    turn_rate = rng.choice([-0.05, -0.02, 0.0, 0.02, 0.05])
                    cat_name = "human.pedestrian.adult"
                else:
                    speed = rng.uniform(4.0, 16.0) # Vehicle speed (m/s)
                    turn_rate = rng.choice([-0.015, -0.008, 0.0, 0.008, 0.015])
                    cat_name = "vehicle.car"

                heading = rng.uniform(0.0, 2.0 * math.pi)

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

                # Neighbors
                neighbors = []
                num_neighbors = rng.randint(1, 4)
                for n in range(num_neighbors):
                    nx = past_traj[0][0] + rng.uniform(-15.0, 15.0)
                    ny = past_traj[0][1] + rng.uniform(-15.0, 15.0)
                    n_speed = speed + rng.uniform(-2.0, 2.0)
                    n_traj = []
                    for _ in range(20):
                        n_traj.append([round(nx, 3), round(ny, 3)])
                        nx += n_speed * 0.1 * math.cos(heading)
                        ny += n_speed * 0.1 * math.sin(heading)
                    neighbors.append(n_traj)

                # Simulated HD map lanes
                map_lanes = []
                for lane_idx in range(6):
                    lane_offset = (lane_idx - 2.5) * 3.5
                    p1 = [round(past_traj[-1][0] + lane_offset, 2), round(past_traj[-1][1] - 30.0, 2)]
                    p2 = [round(p1[0] + 60.0 * math.cos(heading), 2), round(p1[1] + 60.0 * math.sin(heading), 2)]
                    map_lanes.append([p1, p2])

                items.append({
                    "instance_token": f"syn_{split}_{i:06d}",
                    "category": cat_name,
                    "city": cities[i % len(cities)],
                    "past_traj": past_traj,
                    "future_traj": future_traj,
                    "neighbors_past": neighbors,
                    "map_lanes": map_lanes
                })

            rng.shuffle(items)
            return items

        val_count = max(800, count // 4)
        train_items = generate_collection(count, "train")
        val_items = generate_collection(val_count, "val")
        return train_items, val_items


class SequenceDatasetWriter(IDatasetWriter):
    """
    Saves sequences into train_sequences.json and val_sequences.json with deterministic shuffling.
    Reports accurate file paths and disk sizes in MB.
    """
    def write(self, train_items: List[Dict[str, Any]], val_items: List[Dict[str, Any]], output_dir: str, seed: int = 42) -> Tuple[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, "train_sequences.json")
        val_path = os.path.join(output_dir, "val_sequences.json")

        rng = random.Random(seed)
        rng.shuffle(train_items)
        rng.shuffle(val_items)

        print("=" * 80)
        print(f"[Dataset Writer] Saving {len(train_items):,} train sequences to: {train_path}")
        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(train_items, f)
        size_tr = os.path.getsize(train_path) / (1024 * 1024)
        print(f" Saved Train Sequences:{train_path} ({size_tr:.2f} MB)")
        print("-" * 80)
        print(f"[Dataset Writer] Saving {len(val_items):,} val sequences to: {val_path}")
        with open(val_path, "w", encoding="utf-8") as f:
            json.dump(val_items, f)
        size_va = os.path.getsize(val_path) / (1024 * 1024)
        print(f" Saved Val Sequences:  {val_path} ({size_va:.2f} MB)")
        print("=" * 80)


        return train_path, val_path



# ==============================================================================
# Pipeline Coordinator (Dependency Inversion Principle - DIP)
# ==============================================================================

class DataPreparationPipeline:
    """
    Orchestrates dataset verification, map-expansion checking, and extraction.
    Follows Dependency Inversion: accepts interfaces for validator, extractors, and writer.
    Maintains DataConfig as the Single Source of Truth for all execution parameters.
    """
    def __init__(
        self,
        config: Optional[DataConfig] = None,
        map_validator: Optional[IMapExpansionValidator] = None,
        real_extractor: Optional[BaseTrajectoryExtractor] = None,
        synthetic_generator: Optional[BaseTrajectoryExtractor] = None,
        dataset_writer: Optional[IDatasetWriter] = None
    ):
        self.config = config or DataConfig()
        self.map_validator = map_validator or NuScenesMapExpansionValidator()
        self.real_extractor = real_extractor or NuScenesRealDataExtractor(config=self.config)
        self.synthetic_generator = synthetic_generator or SyntheticTrajectoryGenerator()
        self.dataset_writer = dataset_writer or SequenceDatasetWriter()

    def prepare(
        self,
        dataroot: Optional[str] = None,
        output_dir: Optional[str] = None,
        target_count: Optional[int] = -999,
        mode: Optional[str] = None,
        seed: Optional[int] = None,
        category: Optional[Union[str, List[str]]] = None,
        max_train_scenes: Optional[int] = None,
        max_val_scenes: Optional[int] = None,
        config: Optional[DataConfig] = None
    ) -> bool:
        """
        Executes preparation workflow:
          - Merges parameters from DataConfig instance as the primary Source of Truth.
          - If mode == 'real': enforces real extraction.
          - If mode == 'synthetic': enforces synthetic generation.
          - If mode == 'auto': checks if maps/expansion exists on disk.
            Extracts real data only if maps/expansion is available; otherwise falls back.
          - If target_count is None: extracts ALL available sequences without limitation.
        """
        active_cfg = config or self.config
        active_dataroot = dataroot if dataroot is not None else active_cfg.dataroot
        active_output_dir = output_dir if output_dir is not None else active_cfg.output_dir
        active_count = target_count if target_count != -999 else active_cfg.target_count
        active_mode = mode if mode is not None else active_cfg.mode
        active_seed = seed if seed is not None else active_cfg.seed
        active_category = category if category is not None else active_cfg.category
        active_train_scenes = max_train_scenes if max_train_scenes is not None else active_cfg.max_train_scenes
        active_val_scenes = max_val_scenes if max_val_scenes is not None else active_cfg.max_val_scenes

        count_label = f"{active_count:,} shuffled sequences" if active_count is not None else "ALL available sequences (No truncation)"
        cat_label = str(active_category) if active_category is not None else "All Categories"

        # 1. Force Synthetic Mode
        if active_mode == "synthetic":
            print("[Pipeline Policy] Forced SYNTHETIC mode requested.")
            train_items, val_items = self.synthetic_generator.extract_sequences(
                dataroot=active_dataroot,
                target_count=active_count,
                seed=active_seed,
                category=active_category
            )
            tr_p, va_p = self.dataset_writer.write(train_items, val_items, active_output_dir, seed=active_seed)
            print_dataset_summary(train_items, val_items, tr_p, va_p, active_category)
            return True

        # 2. Check Map Expansion & Raw Files
        maps_available, map_dir, map_msg = self.map_validator.is_map_expansion_available(active_dataroot)
        raw_meta_exists = os.path.exists(os.path.join(active_dataroot, "v1.0-trainval"))

        has_devkit = False
        try:
            import nuscenes
            has_devkit = True
        except ImportError:
            pass

        # 3. Decision Logic
        should_use_real = (active_mode == "real") or (active_mode == "auto" and maps_available and raw_meta_exists and has_devkit)

        if should_use_real:
            print(f"[Pipeline Policy] maps/expansion verified ({map_msg}). \nExecuting REAL nuScenes extraction ({count_label} | Category: {cat_label})...")
            try:

                train_items, val_items = self.real_extractor.extract_sequences(
                    dataroot=active_dataroot,
                    target_count=active_count,
                    seed=active_seed,
                    category=active_category,
                    max_train_scenes=active_train_scenes,
                    max_val_scenes=active_val_scenes
                )
                tr_p, va_p = self.dataset_writer.write(train_items, val_items, active_output_dir, seed=active_seed)
                print_dataset_summary(train_items, val_items, tr_p, va_p, active_category)
                return True
            except Exception as e:
                print(f"[Pipeline Warning] Real extraction encountered an error: {e}")
                if active_mode == "real":
                    raise e
                print("[Pipeline Fallback] Falling back to high-capacity synthetic generator...")

        # Fallback when maps/expansion is not available
        print(f"[Pipeline Notice] maps/expansion not accessible ({map_msg}).")
        fallback_count = active_count if active_count is not None else 6400
        print(f"Generating {fallback_count:,} uniformly shuffled synthetic sequences for reliable development...")
        train_items, val_items = self.synthetic_generator.extract_sequences(
            dataroot=active_dataroot,
            target_count=fallback_count,
            seed=active_seed,
            category=active_category
        )
        tr_p, va_p = self.dataset_writer.write(train_items, val_items, active_output_dir, seed=active_seed)
        print_dataset_summary(train_items, val_items, tr_p, va_p, active_category)
        return True


# ==============================================================================
# Backward-Compatible Function Facades
# ==============================================================================

def run_official_nuscenes_extraction(
    dataroot: str,
    output_dir: Optional[str] = None,
    target_count: Optional[int] = 6400,
    max_train_scenes: Optional[int] = None,
    max_val_scenes: Optional[int] = None,
    category: Optional[Union[str, List[str]]] = None
) -> bool:
    """Backward-compatible facade for official real extraction."""
    pipeline = DataPreparationPipeline(
        real_extractor=NuScenesRealDataExtractor(
            max_train_scenes=max_train_scenes,
            max_val_scenes=max_val_scenes,
            category=category
        )
    )
    return pipeline.prepare(
        dataroot=dataroot,
        output_dir=output_dir,
        target_count=target_count,
        mode="real",
        category=category,
        max_train_scenes=max_train_scenes,
        max_val_scenes=max_val_scenes
    )


def run_synthetic_preparation(output_dir: str, count: Optional[int] = 6400, category: Optional[Union[str, List[str]]] = None) -> bool:
    """Backward-compatible facade for synthetic preparation."""
    pipeline = DataPreparationPipeline()
    return pipeline.prepare(dataroot="", output_dir=output_dir, target_count=count, mode="synthetic", category=category)


def main():
    default_cfg = DataConfig()

    def parse_optional_count(val):
        if val is None or str(val).strip().lower() in ("none", "null", "all", "-1"):
            return None
        return int(val)

    parser = argparse.ArgumentParser(description="TrajectoryLNAS Data Preparation Pipeline (SOLID)")
    parser.add_argument("--dataroot", type=str, default=default_cfg.dataroot, help="Path to raw nuScenes root")
    parser.add_argument("--output_dir", type=str, default=default_cfg.output_dir, help="Destination directory for sequences JSON")
    parser.add_argument("--mode", type=str, default=default_cfg.mode, choices=["auto", "real", "synthetic"])
    parser.add_argument(
        "--count",
        type=parse_optional_count,
        default=default_cfg.target_count,
        help="Sequence count (e.g., 6400). Pass 'none' or 'all' to extract ALL available sequences without limitation."
    )
    parser.add_argument("--category", type=str, default=default_cfg.category, help="Filter agent category (e.g. 'pedestrian', 'vehicle', 'car')")
    parser.add_argument("--seed", type=int, default=default_cfg.seed, help="Random seed for deterministic shuffling")
    parser.add_argument("--max_train_scenes", type=int, default=default_cfg.max_train_scenes, help="Limit number of train scenes")
    parser.add_argument("--max_val_scenes", type=int, default=default_cfg.max_val_scenes, help="Limit number of val scenes")
    args = parser.parse_args()

    pipeline = DataPreparationPipeline(config=default_cfg)
    pipeline.prepare(
        dataroot=args.dataroot,
        output_dir=args.output_dir,
        target_count=args.count,
        mode=args.mode,
        seed=args.seed,
        category=args.category,
        max_train_scenes=args.max_train_scenes,
        max_val_scenes=args.max_val_scenes
    )


if __name__ == "__main__":
    main()
