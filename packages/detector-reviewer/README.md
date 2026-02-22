# detector-reviewer

Visual QA tool to compare model inference against ground truth labels.

## Features

- Auto-load latest `best.pt` from `artifacts/detector-train/latest_run.json`
- Run inference on selected samples
- Overlay GT (green) and predictions (red)
- Filter by split and confidence threshold
- Navigate samples quickly to inspect failures
- Strict OBB-only labels and model outputs (fails on bbox formats)

## Run

```bash
uv run detector-reviewer --dataset coco128 --datasets-base-root dataset/augmented
```

Optional explicit weights:

```bash
uv run detector-reviewer --dataset-root dataset/augmented/coco128 --weights path/to/best.pt
```
