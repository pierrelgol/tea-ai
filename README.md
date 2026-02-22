# tea-ai

Linear YOLO OBB fine-tuning pipeline.

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

Cleanup:

```bash
just clean
just fclean
```
