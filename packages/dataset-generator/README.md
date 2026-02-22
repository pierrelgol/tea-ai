# dataset-generator

Geometry-pure synthetic dataset generator for compositing labeled targets onto COCO8 backgrounds.

## Run

```bash
uv run dataset-generator --seed 42
```

## Output

- `dataset/augmented/images/train|val`
- `dataset/augmented/labels/train|val`
- `dataset/augmented/meta/train|val`
- `dataset/augmented/classes.txt`
- `dataset/augmented/classes_map.json`

Labels are YOLO OBB (`class x1 y1 x2 y2 x3 y3 x4 y4`, normalized) derived directly from projected canonical corners using the same homography used for image warping.
