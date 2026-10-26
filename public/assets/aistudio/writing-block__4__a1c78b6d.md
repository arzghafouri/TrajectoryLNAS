## Important Clarification: Development Subset vs. Final Research Dataset

The current 50-scene dataset is intentionally created through scene-level subsampling. It is a temporary development and testing subset, not the final dataset intended for the research experiments.

### 1. Development and Testing Stage

The current dataset contains:

- 40 training scenes.
- 10 validation scenes.
- 22,562 extracted trajectory samples.
- Strict scene-level separation.
- 100% inspection of the extracted subset.

Its purpose is to accelerate development and testing, including:

- Verifying the extraction and preprocessing pipeline.
- Debugging the Dataset and DataLoader.
- Testing model forward and backward passes.
- Identifying tensor shape and masking problems.
- Evaluating candidate operators.
- Testing the NAS implementation.
- Comparing preliminary architecture configurations.
- Identifying memory and computational bottlenecks.

Results obtained from this subset must be treated as preliminary development results, not final research results.

### 2. Final Research Dataset

The intended final experiments will use a substantially larger nuScenes subset containing approximately 1,000 scenes.

The final dataset must be processed using the verified and corrected pipeline established during development.

The audit must determine whether the current preprocessing and model architecture can scale from 50 scenes to approximately 1,000 scenes without introducing:

- Data leakage.
- Memory bottlenecks.
- Excessive preprocessing time.
- Incorrect scene associations.
- Invalid map-coordinate transformations.
- Data distribution inconsistencies.

Do not assume that successful execution on 50 scenes guarantees correct or efficient execution on 1,000 scenes.

### 3. NAS Search Sample Percentage

The `configured_search_samples_percent` parameter is a configurable runtime variable, not a fixed dataset limitation.

Its current development value is 20%.

During final architecture search, this parameter will be set to 100%, meaning that all eligible training samples in the final training split should be available to the NAS search process.

Verify the actual implementation of this parameter and determine:

1. Whether it controls the number of samples used for network-weight optimization.
2. Whether it controls the samples used for architecture optimization.
3. Whether it is applied to the training split only.
4. Whether validation data are incorrectly included in training.
5. Whether setting it to 100 actually makes all eligible training samples accessible.
6. Whether the configured percentage is applied before or after category-aware sampling.
7. Whether the sampling process remains deterministic and reproducible.

Important distinction:

- 20% means a reduced training subset for faster development experiments.
- 100% means the complete eligible training split is used during final NAS search.
- Validation samples must remain separate from network-weight training.
- An independent final test set must not participate in architecture optimization or hyperparameter selection.

### 4. Two-Stage Experimental Strategy

Assess the project using the following two-stage research strategy.

**Stage A — Development and Preliminary Search**

Use the current 50-scene subset to verify implementation correctness, identify unsuitable operators, and conduct preliminary architecture experiments.

The purpose is to reduce implementation risk and eliminate clearly unsuitable configurations.

Do not claim final architectural superiority from this stage.

**Stage B — Final Research Experiments**

Use approximately 1,000 scenes for the final research dataset.

Perform the final training and architecture experiments using the complete eligible training split, with `configured_search_samples_percent = 100`.

Use scene-level separation for training, validation, and independent testing.

The validation set may be used for architecture selection, early stopping, and hyperparameter tuning. The independent test set must be reserved for final evaluation.

If the final dataset contains approximately 1,000 scenes in total, recommend a suitable scene-level train/validation/test allocation based on the available scene distribution.

### 5. Architecture Transferability

Investigate whether architecture choices made using the 50-scene development subset are likely to remain appropriate when scaling to approximately 1,000 scenes.

Pay particular attention to:

- Operator ranking changes with dataset size.
- Sensitivity to scene diversity.
- Sensitivity to category distribution.
- Interaction graph density.
- HD Map topology diversity.
- Architecture parameter stability.
- Generalization across scenes and geographic locations.

Recommend which preliminary experiments can be performed on 50 scenes and which conclusions must be revalidated on the final dataset.

### 6. Required Audit Conclusion

Explicitly distinguish three types of results:

1. **Development verification:** Evidence that the implementation works correctly on the 50-scene subset.
2. **Preliminary architecture assessment:** Evidence about candidate operators and architecture choices from reduced-scale experiments.
3. **Final research evaluation:** Results obtained using the approximately 1,000-scene dataset and an independent test set.

Do not mix these three result categories in the final research report.

The current 50-scene dataset must be evaluated according to its intended role as a development subset, not criticized merely because it is smaller than the final research dataset.