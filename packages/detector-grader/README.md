# detector-grader

Canonical evaluation and grading tool for detector runs.

## What it does

- Resolves a model source (`latest`, `.`, weights path, run dir, or prediction key)
- Optionally runs inference via `detector-infer`
- Loads GT + prediction OBB labels and scores geometric quality
- Produces per-sample, per-split, and run-level grading reports
- Exposes evaluation-like summary metrics for training loops

## Run

```bash
uv run detector-grader \
  --dataset coco128 \
  --datasets-base-root dataset/augmented \
  --predictions-root predictions \
  --model latest
```
