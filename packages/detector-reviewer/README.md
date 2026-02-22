# detector-reviewer

Visual QA tool for ground truth vs existing prediction labels.

## Features

- Loads dataset samples and prediction files for a selected model key
- Overlays GT (green) and predictions (red)
- Filter by split and confidence threshold
- Fast sample navigation for failure inspection

## Run

```bash
uv run detector-reviewer --dataset coco128 --model latest
```
