# detector-infer

Focused inference module.

Responsibilities:
- Load weights
- Run inference
- Save predictions in YOLO format
- Enforce OBB-only predictions (fails if model does not output OBB)

## Run

```bash
uv run detector-infer \
  --weights artifacts/detector-train/runs/my-run/weights/best.pt \
  --dataset coco8 \
  --datasets-base-root dataset/augmented \
  --output-root predictions \
  --model-name yolo11n-obb
```
