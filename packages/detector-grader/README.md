# detector-grader

Canonical evaluation and grading tool for detector runs.

## What it does

- Resolves model/weights from shared config
- Optionally runs inference via `detector-infer`
- Loads GT + prediction OBB labels and scores geometric quality
- Produces per-sample, per-split, and run-level grading reports
- Exposes evaluation-like summary metrics for training loops

## Run

```bash
uv run detector-grader --config config.json
```
