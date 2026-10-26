## Final Objective: Dataset-Driven Neural Architecture Alignment

The ultimate objective of this audit is not merely to evaluate the existing operators independently. It is to establish a technically coherent and dataset-driven neural architecture in which every operator is explicitly aligned with the information available in the dataset.

For each architecture branch, perform the following:

1. Identify the exact dataset features available.
2. Identify which features the current operators actually consume.
3. Identify information that is available but currently unused or discarded.
4. Determine the mathematical operations required to exploit that information.
5. Select or design operators that are compatible with the actual tensor representations.
6. Verify that the outputs of all operators can be integrated into the corresponding mixed-operation branch.
7. Evaluate whether the proposed operators provide complementary rather than redundant capabilities.

### Mandatory Alignment Matrix

Produce a final mapping:

| Dataset Feature | Tensor Representation | Current Operator | Identified Gap | Recommended Operator | Required Modification |
|---|---|---|---|---|---|

### Architecture-Level Consistency

After evaluating all branches, verify that the complete architecture forms a coherent information-processing pipeline:

- Temporal branch: Extract meaningful motion representations from the target agent's history.
- Interaction branch: Model spatial and temporal dependencies between the target and neighboring agents.
- HD Map branch: Encode lane geometry, orientation, topology, and agent-to-map relationships.
- Fusion stage: Combine complementary information without losing important spatial or temporal relationships.
- Prediction head: Generate valid future trajectories in the intended coordinate system.

### Final Alignment Requirements

The recommended architecture must satisfy all of the following:

- Every input feature has a clearly defined purpose.
- Every operator has a justified role.
- No important dataset information is silently discarded.
- Tensor dimensions, coordinate systems, and feature semantics remain consistent.
- Missing modalities are handled safely.
- Redundant operators are eliminated or experimentally justified.
- Computational complexity is appropriate for the intended hardware.
- The architecture can scale from the 50-scene development subset to the approximately 1,000-scene final dataset.

The final deliverable must include a proposed integrated architecture, not merely a list of independently recommended operators.

However, do not modify the source code during this audit. First provide the complete architecture alignment specification, including tensor shapes, mathematical definitions, data-flow diagrams, and implementation requirements. Wait for explicit approval before implementing the proposed changes.