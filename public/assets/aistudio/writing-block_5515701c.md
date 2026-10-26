# Task: Implement an Independent Dataset Statistics and Reporting Tool

Act as a Senior Machine Learning Engineer specializing in PyTorch, autonomous driving, trajectory prediction, and the nuScenes dataset.

Repository:\
[https://github.com/arzghafouri/TrajectoryLNAS/tree/main/python_src](https://github.com/arzghafouri/TrajectoryLNAS/tree/main/python_src)

Implement a comprehensive, independent dataset statistics and reporting tool for the existing TrajectoryLNAS project.

## 1. Mandatory Architecture and File Requirements

Create the following file:

`python_src/data/data_summery.py`

Inside it, implement the main class:

`DataSummery`

**CRITICAL REQUIREMENTS:**

1. `data_summery.py` must be independently executable.
2. It must NOT require running `main.py`.
3. Do NOT integrate its execution into `main.py`.
4. Do NOT modify the training or DARTS search pipeline to invoke this reporting tool.
5. All dataset inspection, statistics calculation, validation, aggregation, and report generation must be implemented inside the `DataSummery` class in `data_summery.py`.
6. Other existing project files should remain unchanged unless a change is absolutely necessary. Prefer zero modifications to existing files.
7. The script must load and inspect the actual dataset using the existing project dataset classes and preprocessing logic.
8. Do not create a separate synthetic dataset or a parallel data-loading implementation that differs from the actual project pipeline.
9. Handle Python imports correctly because the script is now located inside the `data` subdirectory.

The intended execution command from the repository root is:

```powershell
python python_src/data/data_summery.py
```

The script must also support execution from the `python_src` directory:

```powershell
python data/data_summery.py
```

## 2. Independent Dataset Loading

Inspect the existing project and identify how the real nuScenes dataset is loaded.

Reuse the existing:

- Dataset classes.
- Dataset configuration.
- Dataset preprocessing.
- nuScenes metadata and annotations.
- Temporal trajectory extraction.
- Surrounding-agent extraction.
- HD Map extraction.
- Category labels.
- Train/validation splitting logic.

The reporting tool must independently initialize the dataset without requiring `main.py`.

Do not duplicate the entire training pipeline.

Avoid importing `main.py` merely to access dataset objects.

Do not trigger DARTS search, model construction, model training, checkpoint loading, or GPU allocation.

Prefer CPU-only dataset inspection.

If the existing dataset loader requires configuration, initialize it directly using the project's configuration classes.

If the dataset path is not configured, allow it to be supplied through a command-line argument.

Example:

```powershell
python python_src/data/data_summery.py --data-root "F:\Program\Dataset\nuScenes"
```

Inspect the repository first and adapt the arguments to the actual dataset-loading implementation.

## 3. Command-Line Interface

Implement an independent CLI using `argparse`.

Support the following options where applicable:

```powershell
python python_src/data/data_summery.py
```

```powershell
python python_src/data/data_summery.py --data-root "F:\Program\Dataset\nuScenes"
```

```powershell
python python_src/data/data_summery.py --mode full
```

```powershell
python python_src/data/data_summery.py --mode sampled --sample-percent 20
```

```powershell
python python_src/data/data_summery.py --output-dir "python_src/data/report"
```

```powershell
python python_src/data/data_summery.py --format txt json csv
```

Suggested arguments:

- `--data-root`: Dataset root directory.
- `--mode`: `full` or `sampled`.
- `--sample-percent`: Percentage of samples to inspect in sampled mode.
- `--seed`: Random seed for reproducible inspection.
- `--output-dir`: Report output directory.
- `--format`: One or more output formats: `txt`, `json`, `csv`.
- `--dataset-version`: nuScenes version, if needed.
- `--split`: `train`, `val`, or `both`.

Use actual project configuration defaults when available.

Do not invent configuration values.

## 4. Output Directory and Report Files

All generated reports must be saved inside:

`python_src/report/`

This output directory must be resolved relative to the location of `data_summery.py`, not the current working directory.

Use `pathlib.Path(__file__).resolve().parent` to determine the script directory.

Create the report directory automatically if it does not exist.

Generate timestamped report filenames to avoid overwriting previous reports.

Recommended project structure:

```text
python_src/
    data/
        data_summery.py
        report/
            dataset_report_YYYYMMDD_HHMMSS.txt
            dataset_report_YYYYMMDD_HHMMSS.json
            dataset_report_YYYYMMDD_HHMMSS.csv
```

Output requirements:

### TXT

A readable, well-formatted report containing all sections and statistics.

### JSON

A structured representation of all statistics, including metadata, units, inspection configuration, warnings, and validation results.

The JSON must be valid and machine-readable.

Represent unavailable values as `null`, not fabricated zeros.

### CSV

A structured tabular export of the key metrics.

Use clear column names, section names, metric names, values, and units.

Do not flatten unrelated information in a way that makes it ambiguous.

The TXT, JSON, and CSV files must represent the same inspection run and consistent statistics.

At the end of execution, print a short terminal message indicating:

- Dataset inspection status.
- Number of inspected samples.
- Report output directory.
- Exact paths of generated report files.

The complete report must be saved to disk even if terminal output is disabled.

## 5. DataSummery Class

Implement the following class inside `data_summery.py`.

Suggested constructor:

```python
class DataSummery:
    def __init__(
        self,
        data_root=None,
        config=None,
        output_dir=None,
        mode="full",
        sample_percent=100,
        seed=42,
        split="both"
    ):
        ...
```

Adapt this interface to the actual repository.

Required methods:

```python
load_dataset()
inspect_dataset()
collect_statistics()

analyze_temporal_sequences()
analyze_interaction_sequences()
analyze_hd_map_sequences()

analyze_categories()
analyze_search_samples()
analyze_data_quality()
analyze_train_val_split()

validate_dataset()

generate_txt_report()
generate_json_report()
generate_csv_report()

save_reports()
run()
```

All these methods must belong to `DataSummery`.

The `run()` method must orchestrate the complete workflow:

1. Resolve configuration and dataset paths.
2. Load the actual dataset.
3. Inspect the dataset.
4. Collect statistics.
5. Validate data quality.
6. Generate all requested reports.
7. Save reports to the output directory.
8. Print the report locations and execution summary.

Do not place reporting logic in other files.

## 6. Required Dataset Report Sections

### [1] DATASET INFORMATION

- Dataset Name.
- Dataset Source.
- Dataset Root.
- Dataset Version.
- Dataset Type: Real / Synthetic / Mixed.
- Real Samples.
- Synthetic Samples.
- Fallback Activated.
- Fallback Reason.
- Total Samples.
- Training Samples.
- Validation Samples.
- Train/Validation Ratio.
- Inspection Mode.
- Number of Inspected Samples.
- Inspection Percentage.
- Random Seed.

### [2] SEQUENCE AVAILABILITY

| Sequence Type         | Available Samples | Percentage |
| --------------------- | ----------------: | ---------: |
| Temporal Sequences    |                   |            |
| Interaction Sequences |                   |            |
| HD Map Sequences      |                   |            |
| Complete Sequences    |                   |            |

Count actual valid data, not just the existence of dictionary keys.

### [3] TEMPORAL SEQUENCE STATISTICS

Display:

- Average Past Timesteps.
- Average Future Timesteps.
- Minimum and Maximum Past Timesteps.
- Minimum and Maximum Future Timesteps.
- Average Past Duration.
- Average Future Duration.
- Average Sampling Interval.
- Sampling Interval Standard Deviation.
- Average Past Trajectory Length.
- Average Future Trajectory Length.
- Average Past Displacement.
- Average Future Displacement.
- Average Target-Agent Speed.
- Average Target-Agent Acceleration.
- Percentage of Valid Past Trajectories.
- Percentage of Valid Future Trajectories.

Use actual timestamps when available.

If duration is estimated from a configured sampling interval, explicitly label it as estimated.

Calculate trajectory length from consecutive valid positions.

Calculate displacement from the first and last valid positions.

### [4] INTERACTION SEQUENCE STATISTICS

Display:

- Average Neighbors/Agent.
- Minimum Neighbors/Agent.
- Maximum Neighbors/Agent.
- Average Valid Neighbor Timesteps.
- Average Neighbor-Agent Distance.
- Average Relative Speed.
- Average Valid Neighbors per Sequence.
- Percentage of Samples with at Least One Neighbor.
- Percentage of Samples with No Neighbors.
- Neighbor Category Distribution.

Count only valid surrounding agents.

Exclude padding and the target agent itself.

### [5] HD MAP SEQUENCE STATISTICS

Display:

- Average HD Map Lanes.
- Minimum HD Map Lanes.
- Maximum HD Map Lanes.
- Average Points per Lane.
- Minimum Points per Lane.
- Maximum Points per Lane.
- Average Total Map Points per Sequence.
- Average Lane Length.
- Average Distance to Nearest Lane.
- Percentage of Samples with Valid HD Map.
- Percentage of Samples without HD Map.
- Lane Category Distribution.

Count actual valid lane polylines and their points.

Do not treat empty or invalid map tensors as valid map data.

Verify coordinate compatibility before computing distances.

Explicitly report why map data is missing if extraction or loading has failed.

### [6] CATEGORY DISTRIBUTION

| Category | Training Count | Training % | Validation Count | Validation % | Total |
| -------- | -------------: | ---------: | ---------------: | -----------: | ----: |

Include all categories found in actual data.

Also report unknown, missing, and invalid labels.

### [7] SEARCH CONFIGURATION AND SAMPLE SELECTION

Display:

- Configured Search Samples Percent.
- Total Available Training Samples.
- Expected Search Samples.
- Actual Selected Search Samples.
- Actual Search Samples Percentage.
- Validation Samples Used During Search.
- Sampling Strategy.
- Random Seed.

Inspect the actual project implementation of `search_samples_percent`.

Do not confuse the configured percentage with the actual selected number.

If the search subset cannot be reproduced independently without invoking the training pipeline, report the configured value and clearly mark the actual selected count as unavailable. Do not invent it.

### [8] DATA QUALITY AND VALIDATION

Display:

- Total Samples Inspected.
- Valid Samples.
- Invalid Samples.
- Missing Temporal Data.
- Missing Interaction Data.
- Missing HD Map Data.
- NaN Values.
- Infinite Values.
- Invalid Coordinates.
- Empty Trajectories.
- Invalid Trajectory Lengths.
- Duplicate Samples.
- Invalid Category Labels.
- Invalid Masks.
- Filtered Samples.

For every issue, report count, percentage, field, and reason.

Distinguish missing data from invalid data.

Do not silently filter samples.

Do not classify legitimate zero coordinates as invalid.

### [9] TRAIN / VALIDATION SPLIT SUMMARY

Display:

- Total Dataset Samples.
- Training Samples.
- Validation Samples.
- Training Percentage.
- Validation Percentage.
- Temporal Data Availability by Split.
- Interaction Data Availability by Split.
- HD Map Data Availability by Split.
- Category Distribution by Split.
- Duplicate Samples Across Splits.
- Scene Leakage Status.

Determine whether the project uses sample-level or scene-level splitting.

### [10] FINAL DATASET SUMMARY

Display:

- Dataset Status: READY / WARNING / FAILED.
- Total Samples.
- Training Samples.
- Validation Samples.
- Temporal Data Availability.
- Interaction Data Availability.
- HD Map Data Availability.
- Complete Valid Samples.
- Invalid Samples.
- Filtered Samples.
- Synthetic Data Detected.
- Fallback Activated.
- Important Warnings.

## 7. Accuracy Requirements

All reported statistics must be derived from actual dataset samples.

Never use hardcoded, random, illustrative, or assumed statistics.

If a metric cannot be computed, use `N/A` in TXT, `null` in JSON, and an explicit unavailable status in CSV.

Do not interpret padded zeros as actual agents or map coordinates.

Use actual validity masks.

Do not assume fixed temporal lengths without verifying the data.

Do not modify the dataset during inspection.

Prefer CPU-based inspection.

Support:

- Full inspection.
- Reproducible sampled inspection.

Default to full inspection where computationally reasonable.

Clearly label sampled statistics as estimates.

## 8. Synthetic Data and Fallback

Inspect existing synthetic/mock dataset mechanisms.

The tool must distinguish real, synthetic, and mixed data.

Do not silently switch to synthetic data if real nuScenes loading fails.

If real-data loading fails:

- Report the original exception.
- Preserve traceback in diagnostic output or error logs.
- Mark dataset status as FAILED.
- Do not generate a misleading successful report.

If synthetic data is explicitly enabled, report it clearly.

## 9. Execution and Independence

The tool must work independently from the training system.

It must NOT:

- Import or execute `main.py`.
- Start DARTS search.
- Train any model.
- Load model checkpoints.
- Allocate GPU memory unnecessarily.
- Modify training configurations.
- Modify dataset files.
- Change random states used by training.

Use the project's existing dataset implementation directly.

Make sure the script can be executed from the repository root and from the `python_src` directory.

Handle imports correctly from its new location inside `python_src/data/`.

Do not hardcode machine-specific paths.

The default report output path must always resolve to `python_src/report/`, regardless of the current working directory.

## 10. Testing

Test the script with:

1. Valid real nuScenes samples, if accessible.
2. Samples with valid temporal data.
3. Samples without neighbors.
4. Samples without HD Map.
5. Malformed tensors.
6. NaN and Inf values.
7. Empty trajectories.
8. Synthetic/mock data.
9. Duplicate sample identifiers.
10. Train/validation leakage.
11. TXT, JSON, and CSV output generation.
12. Independent execution without `main.py`.
13. Execution from both repository root and `python_src`.
14. Correct creation of `python_src/report/`.

Do not fabricate test results if the real dataset is unavailable.

## 11. Final Deliverables

Provide:

1. Complete `python_src/data/data_summery.py` implementation.
2. Explanation of the `DataSummery` class.
3. Exact commands to run the tool.
4. Exact output directory and generated filenames.
5. Example of the final terminal completion message.
6. Test results and verification status.
7. Remaining limitations.

**FINAL ACCEPTANCE CRITERIA:**

- The file is located at `python_src/data/data_summery.py`.
- `DataSummery` contains all reporting and statistics logic.
- The script is independently executable.
- No invocation is added to `main.py`.
- Reports are automatically saved inside `python_src/report/`.
- TXT, JSON, and CSV outputs contain consistent statistics.
- All metrics are based on actual loaded dataset content.
- The tool does not train models or execute architecture search.
