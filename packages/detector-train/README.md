# detector-train

Lean detector training pipeline.

Responsibilities:
- Write `data.yaml`
- Train YOLO OBB with fixed tuned defaults
- Save weights + summaries
- Log to W&B (auto online/offline fallback)
- Run periodic grader checks (inference + scoring)

## Run

```bash
uv run detector-train --dataset coco8
```
