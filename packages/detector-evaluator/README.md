# detector-evaluator

Pure evaluation package for synthetic/real detection datasets.

## Features

- Ground truth + prediction loading (YOLO bbox/OBB)
- Confidence filtering
- One-to-one IoU matching (greedy)
- Detection metrics + AP@IoU
- Sequential stability metrics
- Geometry metrics from metadata (if available)
- JSON reports + terminal summary
- Optional deterministic visualization artifacts

## Run

```bash
uv run detector-evaluator \
  --dataset coco8 \
  --datasets-base-root dataset/augmented \
  --predictions-root predictions
```
