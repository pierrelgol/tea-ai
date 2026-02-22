# dataset-generator

Geometry-pure synthetic dataset generator for compositing labeled targets onto selected background datasets.

## Run

```bash
uv run dataset-generator --dataset coco8 --seed 42
```

Robust OBB profile (balanced multi-object + photometric complexity):

```bash
uv run dataset-generator \
  --dataset coco128 \
  --output-root dataset/augmented_v2/coco128 \
  --complexity-profile obb_robust_v1 \
  --targets-per-image-min 2 \
  --targets-per-image-max 4 \
  --max-occlusion-ratio 0.45
```

## Output

- `dataset/augmented/<dataset>/images/train|val`
- `dataset/augmented/<dataset>/labels/train|val`
- `dataset/augmented/<dataset>/meta/train|val`
- `dataset/augmented/<dataset>/classes.txt`
- `dataset/augmented/<dataset>/classes_map.json`

Labels are YOLO OBB (`class x1 y1 x2 y2 x3 y3 x4 y4`, normalized) derived directly from projected canonical corners using the same homography used for image warping.

Quality gate utility:

```bash
uv run dataset-generator-gate \
  --integrity-report dataset/augmented_v2/coco128/reports/integrity_report.json \
  --geometry-report dataset/augmented_v2/coco128/reports/geometry_report.json \
  --grade-report dataset/augmented_v2/coco128/grade_reports/grade_report_latest.json \
  --max-geometry-outlier-rate 0.005 \
  --min-run-grade 81.1309
```
