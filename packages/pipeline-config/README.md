# pipeline-config

> Centralized configuration and layout management for the AI Fine-Tuning Pipeline.

`pipeline-config` is the foundational package responsible for parsing, validating, and distributing the global `config.json` payload across all downstream stages of the fine-tuning pipeline. It enforces strict structural contracts, defaults, and typing to ensure reproducible runs.

## Architecture

At its core, `pipeline-config` revolves around two primary data structures:

*   **`PipelineConfig`**: A frozen, typed representation of the global `config.json`. It isolates configuration silos by domain (e.g., `train`, `infer`, `grade`, `generator`), ensuring each pipeline stage only interfaces with its required parameters.
*   **`PipelineLayout`**: A deterministic directory structure resolver. It maps the abstract `run_id` and `model_key` to concrete filesystem paths for artifacts, weights, inference outputs, and evaluations.

```text
+-------------------+       +----------------------+       +-----------------------+
|                   |       |                      |       |                       |
|   config.json     +------>+   load_pipeline_     +------>+   PipelineConfig      |
|                   |       |   config()           |       |   (Domain Silos)      |
+-------------------+       |                      |       +-----------+-----------+
                            +----------------------+                   |
                                                                       v
                                                           +-----------------------+
                                                           |                       |
                                                           |   PipelineLayout      |
                                                           |   (Path Resolution)   |
                                                           |                       |
                                                           +-----------------------+
```

## Features

*   **Strict Schema Validation**: Uses programmatic dict-key checking to ensure no unknown parameters slip into the configuration, failing fast if the schema is violated.
*   **Domain Isolation**: Sub-configurations (`dataset`, `train`, `infer`, etc.) are decoupled, allowing stages like `detector-infer` to remain agnostic to `detector-train` settings.
*   **Deterministic Artifact Layout**: The `PipelineLayout` ensures outputs are consistently mapped to `artifacts/<model_key>/runs/<run_id>/...`, preventing cross-run contamination.
*   **Path Normalization**: Automatically resolves relative paths against the directory containing the configuration file, guaranteeing context-independent execution.

## Usage

```python
from pipeline_config.schema import load_pipeline_config
from pipeline_config.layout import build_layout

# 1. Load and validate configuration
config = load_pipeline_config("path/to/config.json")

# 2. Build the directory layout for the current run
layout = build_layout(
    artifacts_root=config.paths["artifacts_root"],
    model_key=config.run["model_key"],
    run_id=config.run["run_id"]
)

# 3. Access domain-specific config
train_epochs = config.train["epochs"]
infer_device = config.infer["device"]
```
