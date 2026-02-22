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

Labels are YOLO bbox derived from projected canonical corners using the same homography used for image warping.
