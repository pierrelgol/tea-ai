# tea-ai

Linear YOLO OBB fine-tuning pipeline driven by a shared root `config.json`.

## Pipeline

Run stages in order:

```bash
just fetch-dataset
just label-targets
just generate-dataset
just check-dataset
just train
just eval
just review
```

All stages read the same `config.json`.

Cleanup:

```bash
just clean
just fclean
```
