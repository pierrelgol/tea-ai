# detector-train

Lean detector training pipeline.

Responsibilities:
- Write `data.yaml`
- Train YOLO OBB with fixed tuned defaults
- Save weights + summaries
- Log to W&B (auto online/offline fallback)
- Run periodic evaluator checks

## Run

```bash
uv run detector-train --dataset coco8 --datasets-base-root dataset/augmented --artifacts-root artifacts/detector-train
```

Or pass a direct dataset root:

```bash
uv run detector-train --dataset-root dataset/augmented/coco8 --artifacts-root artifacts/detector-train
```
