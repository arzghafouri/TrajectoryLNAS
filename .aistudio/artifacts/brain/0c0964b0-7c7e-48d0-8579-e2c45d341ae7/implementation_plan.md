# TrajectoryLNAS Dataset Preparation Pipeline (prepare_data.py) Redesign Plan

Comprehensive engineering blueprint to refactor `python_src/data/prepare_data.py` into a production-grade, research-ready dataset extraction tool for nuScenes v1.0-trainval motion prediction in TrajectoryLNAS.

---

## User Review & Critical Decisions

> [!IMPORTANT]
> The revised mission specifications enforce the following strict architectural rules:

- **Target Agent Scope**: Strictly moving vehicles (`vehicle.car`, `vehicle.truck`, `vehicle.bus.*`, `vehicle.motorcycle`, `vehicle.bicycle`, `vehicle.trailer`) and pedestrians (`human.pedestrian.*`). Static objects (barriers, traffic cones, bicycle racks) are excluded from target agents but retained in environmental/obstacle context.
- **Timestamp-Based 10Hz Interpolation**: Trajectories are resampled using actual observation timestamps from nuScenes 2Hz keyframe annotations rather than array-index assumptions. Real vs. interpolated points are tracked via validity/interpolation masks.
- **Distance-Sorted Interaction Context**: Up to 15 nearest neighbors within search radius (e.g., 50m) are selected by Euclidean distance at $t=t_{\text{obs}}$, with valid past trajectories and relative position/velocity features.
- **Multi-Layer HD Map Polylines**: Extraction via `NuScenesMap` API queries `lane`, `lane_connector`, `walkway`, `ped_crossing`, and `drivable_area` layers. Polylines store 10 discretized geometric points per lane, sorted by minimum distance to agent.
- **Strict Scene-Level Split & Deduplication**: Training and validation sets partition by `scene_token` (e.g. 700 train / 150 val scenes), guaranteeing zero shared scenes or duplicate `(sample_token, instance_token)` observations.
- **Strict Real Mode Policy**: No silent synthetic fallback in Real mode. Missing map expansion layers or SDK errors produce explicit diagnostic warnings and report statuses.

---

## 1. Overview & Core Concept

- **What It Does**: Refactors `python_src/data/prepare_data.py` into a modular, production-grade pipeline that extracts 10Hz agent trajectories (20 past, 30 future), surrounding interaction neighbors sorted by spatial distance, and vectorized multi-layer HD map polylines from raw nuScenes v1.0-trainval benchmark files.
- **Target Audience**: Researchers and engineers running Neural Architecture Search (DARTS) and trajectory prediction model training in TrajectoryLNAS.
- **Key Value**: Fixes all cross-split duplicates, eliminates scene leakage, resolves 0.0% HD Map polyline availability, extracts true geometry from 4 cities, sorts interaction neighbors by spatial distance, and guarantees strict schema compatibility with `NuScenesPredictionDataset` and `data_summery.py`.

---

## 2. Technical Architecture & Data Strategy

### System Component & Data Flow Architecture

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                         nuScenes Raw Dataset (v1.0-trainval)                   │
│         [v1.0-trainval/*.json, maps/expansion/*.json (4 cities)]             │
└───────────────────────────────────────┬────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                        NuScenesMapExpansionValidator                           │
│   [Validates map expansion layers across Boston Seaport & Singapore cities]    │
└───────────────────────────────────────┬────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                   DataPreparationPipeline Coordinator                          │
│ ┌────────────────────────────────────────────────────────────────────────────┐ │
│ │ Scene-Level Split Manager:                                                 │ │
│ │ - Official nuScenes predict train/val scenes (700 train / 150 val)         │ │
│ │ - Asserts Train Scenes ∩ Val Scenes = ∅ (Zero Scene Leakage)               │ │
│ └─────────────────────────────────────┬──────────────────────────────────────┘ │
│                                       │                                        │
│                                       ▼                                        │
│ ┌────────────────────────────────────────────────────────────────────────────┐ │
│ │ NuScenesRealDataExtractor                                                  │ │
│ │ ├─ Trajectory Extractor (20 past @ 10Hz, 30 future @ 10Hz, timestamps)     │ │
│ │ ├─ Neighbor Interaction Extractor (Distance-sorted, max_neighbors=15)      │ │
│ │ └─ HD Map Polyline Extractor (lane, connector, walkway, crossing)          │ │
│ └─────────────────────────────────────┬──────────────────────────────────────┘ │
│                                       │                                        │
│                                       ▼                                        │
│ ┌────────────────────────────────────────────────────────────────────────────┐ │
│ │ Dataset Quality & Validation Analyzer                                      │ │
│ │ [Checks NaNs, Inf, coords, monotonicity, unique (sample, instance) tokens] │ │
│ └─────────────────────────────────────┬──────────────────────────────────────┘ │
│                                       │                                        │
│                                       ▼                                        │
│ ┌────────────────────────────────────────────────────────────────────────────┐ │
│ │ SequenceDatasetWriter                                                      │ │
│ │ [Atomic JSON streaming writer -> train_sequences.json, val_sequences.json] │ │
│ └────────────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────┬────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                          Downstream Consumers                                  │
│ ├─ NuScenesPredictionDataset (src/data/nuscenes_dataset.py)                    │
│ ├─ DataSummery Reporting Engine (python_src/data/data_summery.py)             │
│ └─ TrajectoryLNAS DARTS Searcher & Final Trainer                               │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Component Specifications

### 3.1 Target Agent Filtering & Exact Matching
- Target agents are validated against explicit, non-ambiguous category categories:
  - Vehicles: `vehicle.car`, `vehicle.truck`, `vehicle.bus.rigid`, `vehicle.bus.bendy`, `vehicle.motorcycle`, `vehicle.bicycle`, `vehicle.trailer`
  - Pedestrians: `human.pedestrian.adult`, `human.pedestrian.child`, `human.pedestrian.construction_worker`, `human.pedestrian.police_officer`, `human.pedestrian.personal_mobility`, `human.pedestrian.stroller`, `human.pedestrian.wheelchair`
- Static objects (`movable_object.barrier`, `movable_object.trafficcone`, `static_object.bicycle_rack`) are strictly filtered out as target agents.

### 3.2 Timestamp-Based 10Hz Trajectory Extraction
- Keyframes in nuScenes are captured at 2Hz (500ms intervals).
- Real observation timestamps ($t_{\text{raw}}$) are retrieved from `sample_annotation` and `PredictHelper`.
- Linear/time-aware spline interpolation maps real observations to 20 past timesteps ($t \in [-2.0s, 0.0s]$ @ 10Hz) and 30 future timesteps ($t \in (0.0s, 3.0s]$ @ 10Hz).
- Trajectory velocity and acceleration are derived from physical displacement over true time intervals $\Delta t$.

### 3.3 Interaction Context & Neighbor Selection
- For target agent $A_{\text{target}}$ at $t=t_{\text{obs}}$ ($t=19$ in past trajectory):
  1. Retrieve all surrounding annotations in the keyframe within search radius $R = 50\text{m}$.
  2. Compute 2D Euclidean distance $d(A_{\text{neighbor}}, A_{\text{target}}) = \sqrt{(x_n - x_t)^2 + (y_n - y_t)^2}$.
  3. Exclude $A_{\text{target}}$ itself.
  4. Sort candidates by $d$ ascending.
  5. Select top 15 nearest valid neighbors (`max_neighbors = 15`).
  6. Extract past trajectories and validity masks for each selected neighbor.

### 3.4 Multi-Layer HD Map Polyline Extraction
- Query `NuScenesMap` API across `boston-seaport`, `singapore-onenorth`, `singapore-queenstown`, and `singapore-hollandvillage`.
- Query layers: `lane`, `lane_connector`, `walkway`, `ped_crossing`, `road_divider`, `drivable_area`.
- For each map record in radius $R=50\text{m}$:
  1. Discretize geometry into 10 points per polyline.
  2. Compute polyline length and minimum distance to target agent position.
  3. Sort polylines by distance to target agent.
  4. Retain up to `max_lanes = 20` polylines per sequence.

### 3.5 Data Splitting & Deduplication
- Partition nuScenes scenes into strict sets:
  - Training scenes: 700 scenes
  - Validation scenes: 150 scenes
- Guarantee $TrainScenes \cap ValScenes = \emptyset$.
- Unique key per sample: `(sample_token, instance_token)`. Eliminate duplicate entries per split.

---

## 4. Contract Schema & Compatibility

Output JSON files (`train_sequences.json`, `val_sequences.json`) preserve the standard schema consumed by `NuScenesPredictionDataset`:
```json
{
  "sample_token": "ca9a28...",
  "instance_token": "bc0192...",
  "scene_token": "fc109...",
  "scene_name": "scene-0001",
  "category": "vehicle.car",
  "city": "boston-seaport",
  "past_traj": [[x0, y0], ... [x19, y19]],
  "future_traj": [[x0, y0], ... [x29, y29]],
  "neighbors_past": [[[x0, y0], ...], ...],
  "map_lanes": [[[x0, y0], ... [x9, y9]], ...]
}
```

---

## 5. Execution & Validation Plan

### CLI Options:
- `--dataroot`: Path to nuScenes dataset root (`F:\Program\Dataset\nuScenes`)
- `--output_dir`: Path to save `train_sequences.json` and `val_sequences.json`
- `--mode`: `real`, `synthetic`, or `auto`
- `--count`: Target count or `all` / `none` for full dataset
- `--category`: Category filter (default: `vehicle,pedestrian`)
- `--seed`: Shuffling seed (default: `42`)
- `--max_train_scenes`, `--max_val_scenes`: Scene limits

### Acceptance Criteria:
1. `prepare_data.py` runs cleanly.
2. HD Map polyline availability is $>0\%$ when map expansion files are present.
3. Cross-split scene leakage and duplicate samples are 0.
4. Output JSON files are loaded successfully by `NuScenesPredictionDataset`.
5. `data_summery.py` executes post-extraction and generates consistent TXT/JSON/CSV reports.
