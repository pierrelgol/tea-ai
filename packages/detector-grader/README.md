# detector-grader

> Canonical evaluation and grading tool for detector runs.

`detector-grader` moves beyond simple mAP metrics to provide a high-precision geometric assessment of Oriented Bounding Box (OBB) detections. It evaluates how well the model has mastered the exact pose, scale, and shape of target objects, producing detailed diagnostic reports used for curriculum generation.

## Architecture

The grader operates as a post-processing stage that consumes ground-truth labels and model predictions, performing greedy matching based on a custom "Quality" metric before calculating deep geometric diagnostics.

```text
 [ Ground Truth OBB ]       [ Model Predictions OBB ]
          |                          |
          v                          v
+-------------------------------------------------------+
|                   DETECTOR GRADER                     |
|                                                       |
|  1. GREEDY MATCHER: Quality-based bipartite matching.  |
|  2. GEOMETRIC SCORER: Five-axis OBB alignment check.   |
|  3. PENALTY ENGINE: FN, FP, and Containment analysis.  |
|  4. REPORT GENERATOR: JSON/CSV artifacts & summaries.  |
+---------------------------+---------------------------+
                            |
                            v
                [ Grading Reports ]
          (Per-sample, Per-split, Run-level)
```

## Technical Core

### 1. Weighted Geometric Scoring
Every matched detection is assigned a score from 0.0 to 1.0 based on a weighted sum of normalized metrics:
*   **IoU Score**: Intersection over Union with optional gamma emphasis to reward high-precision overlaps.
*   **Corner Score**: Exponentially decaying score based on the mean pixel distance between reordered corners.
*   **Angle Score**: Difference in principal axes, weighted by the "orientation reliability" (eccentricity) of the quad.
*   **Center Score**: Normalized Euclidean distance between quad centroids.
*   **Shape Score**: Consistency check of edge length ratios and absolute area ratios.

### 2. Penalty Logic
The final "Grade" for a sample is the mean matched score minus weighted penalties:
*   **False Negative (FN) Penalty**: Heavily penalizes missed targets.
*   **False Positive (FP) Penalty**: Penalizes hallucinations or redundant detections.
*   **Containment Penalty**: Specific penalty for "containment misses" (when a prediction is inside GT but significantly smaller, or vice-versa).

### 3. Diagnostic Metrics
The grader exports over 30 diagnostic fields, including:
*   `angle_le_5_rate`: Percentage of detections with <5° orientation error.
*   `gt_area_missed_ratio`: Quantification of partial occlusions or under-shooting.
*   `hard_class_ids`: Identification of classes consistently failing geometric checks, fed back into the `dataset-generator` curriculum.

## Usage

```bash
uv run detector-grader --config config.json
```

The grader will:
1.  Resolve the latest weights and run inference if predictions are missing.
2.  Normalize all OBB quads to pixel space.
3.  Perform bipartite matching using the quality-weighted greedy algorithm.
4.  Write comprehensive JSON reports to the `eval/` artifact directory.
