# augment-checker

GUI + report checker for synthetic augmentation quality.

## Features

- Dataset integrity checks
- Geometry sanity metrics from metadata (`H`, corners)
- Optional model prediction comparison metrics
- Debug overlay export
- PySide6 visual browser

## Run

```bash
uv run augment-checker --dataset coco8 --datasets-base-root dataset/augmented
```
