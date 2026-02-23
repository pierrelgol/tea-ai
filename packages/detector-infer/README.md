# detector-infer

Focused inference module.

Responsibilities:
- Resolve weights from shared config/latest run
- Run inference
- Save predictions in YOLO format
- Enforce OBB-only predictions (fails if model does not output OBB)

## Run

```bash
uv run detector-infer --config config.json
```
