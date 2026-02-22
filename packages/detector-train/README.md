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

For cleaner live monitoring in W&B:

```bash
uv run detector-train --wandb-log-profile core+diag --wandb-log-every-epoch
```

Hyperparameter knobs can be tuned directly from CLI (examples: `--optimizer AdamW --lr0 0.002 --cos-lr --degrees 2 --scale 0.4`).

Optimization loop (train->grade->compare, with W&B online enforced):

```bash
uv run detector-train-optimize --dataset coco128 --baseline-file baseline.txt
```
