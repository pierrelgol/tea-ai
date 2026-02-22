# detector-grader

CLI scorer for detector runs using strict OBB geometry quality.

## What it does

- Loads GT OBB labels and prediction OBB labels
- Resolves model source from `--model` (`latest`, `.`, weights path, run dir, or predictions key)
- Can run inference automatically before grading
- Matches predictions to GT with polygon IoU
- Computes weighted geometric score components
- Produces per-sample, per-split, and global run grades

## Run

```bash
uv run detector-grader \
  --dataset coco128 \
  --datasets-base-root dataset/augmented \
  --predictions-root predictions \
  --model .
```
