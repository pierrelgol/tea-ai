# dataset-fetcher

Fetches datasets into a top-level `dataset/` directory using versioned profile configs.

## Usage

```bash
uv run dataset-fetcher --dataset coco8
```

Built-in profiles live in `configs/datasets/*.json` (e.g. `coco8`, `coco128`, `coco17`).

By default, this creates/validates `./dataset/<dataset_name>`.
