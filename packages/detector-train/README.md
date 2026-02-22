# detector-train

Focused training module.

Responsibilities:
- Write `data.yaml`
- Launch YOLO training
- Save weights + metrics
- Log to W&B (fail-open, auto offline fallback)

## Run

```bash
uv run detector-train --dataset coco8 --datasets-base-root dataset/augmented --artifacts-root artifacts/detector-train
```
