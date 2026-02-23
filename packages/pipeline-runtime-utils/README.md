# pipeline-runtime-utils

> Cross-cutting runtime abstractions and hardware utilities.

`pipeline-runtime-utils` houses the global utilities required across the fine-tuning pipeline. It provides deterministic behaviors for seeded executions, automatic hardware mapping, and consistent resolution of upstream outputs (like locating the `best.pt` weights from previous stages).

## Architecture

This package abstracts away the "glue code" often repeated in AI pipelines, isolating device logic, seeding, and path resolution.

```text
+-----------------------+      +-----------------------+      +-----------------------+
|                       |      |                       |      |                       |
|   Hardware Context    |      |   Execution Context   |      |   Artifact Resolution |
|   (resolve_device)    |      |   (set_seed)          |      |   (resolve_latest...) |
|                       |      |                       |      |                       |
+-----------+-----------+      +-----------+-----------+      +-----------+-----------+
            |                              |                              |
            v                              v                              v
+-------------------------------------------------------------------------------------+
|                                                                                     |
|                          pipeline-runtime-utils                                     |
|                                                                                     |
+-------------------------------------------------------------------------------------+
```

## Features

*   **Deterministic Execution Seeding**: Provides a unified entry point to initialize `random`, `numpy`, and `torch` (if available) with a synchronized seed.
*   **Automatic Device Resolution**: Normalizes the `"auto"` device configuration by falling back to `cuda` -> `mps` -> `cpu` depending on environment availability, failing gracefully if torch is absent.
*   **Artifact Chaining**: Exposes logic to query the `artifacts_root` directory for the most recent valid trained weights, ensuring smooth integration between the training, inference, and grading stages.

## Usage

```python
from pipeline_runtime_utils.runtime import resolve_device, set_seed
from pipeline_runtime_utils.paths import resolve_latest_weights_from_artifacts

# 1. Establish deterministic behavior
set_seed(42)

# 2. Normalize hardware target
device = resolve_device("auto") # Will return "0", "mps", or "cpu"

# 3. Locate latest training weights
best_weights_path = resolve_latest_weights_from_artifacts(artifacts_root_path)
```
