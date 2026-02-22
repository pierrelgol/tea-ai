# dataset-generator

Geometry-pure synthetic dataset generator for compositing labeled targets onto selected background datasets.

## Run

```bash
uv run dataset-generator --dataset coco8 --seed 42
```

## Output

- `dataset/augmented/<dataset>/images/train|val`
- `dataset/augmented/<dataset>/labels/train|val`
- `dataset/augmented/<dataset>/meta/train|val`
- `dataset/augmented/<dataset>/classes.txt`
- `dataset/augmented/<dataset>/classes_map.json`

Labels are YOLO OBB (`class x1 y1 x2 y2 x3 y3 x4 y4`, normalized) derived directly from projected canonical corners using the same homography used for image warping.
