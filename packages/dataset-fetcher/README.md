# dataset-fetcher

Fetches COCO8 from Ultralytics into a top-level `dataset/` directory.

## Usage

```bash
uv run dataset-fetcher
```

By default, this creates `./dataset/coco8`. If that directory already exists and is non-empty, the command skips re-downloading.
