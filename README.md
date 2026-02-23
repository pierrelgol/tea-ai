# tea-ai

Linear YOLO OBB fine-tuning pipeline driven by a shared root `config.json`.

## Pipeline

Run stages in order:

```bash
just fetch-dataset
just fetch-dinov3
just label-targets
just generate-dataset
just check-dataset
just train
just eval
just review
just profile-pipeline
```

All stages read the same `config.json`.
`profile-pipeline` runs a downscaled end-to-end profiling workflow on `coco128` and reports per-stage hotspots.

Cleanup:

```bash
just clean
just fclean
```
