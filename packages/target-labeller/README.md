# target-labeller

Simple Qt GUI tool to label one bounding box per image and export YOLO format.

## Defaults

- Images: `targets/`
- Labels: `dataset/targets/labels/`
- Class map: `dataset/targets/classes.txt`

## Run

```bash
uv run target-labeller
```

## Controls

- Click + drag on image: draw/replace bounding box
- `Ctrl+S`: save current annotation
- `Left` / `A`: previous image
- `Right` / `D`: next image
- `Finish`: exports labeled samples to:
  - `dataset/targets/images/<class_name>.<ext>` (or `<class_name>_2`, `_3`, ...)
  - `dataset/targets/labels/<class_name>.txt` (same stem as image)
  then closes the tool.
