# pipeline-profile

Profiles the end-to-end tea-ai pipeline on a downscaled representative run.

## Run

```bash
uv run pipeline-profile --config config.json
```

Notes:
- Uses `profile.dataset` (default `coco128`) and `profile.train_epochs` from `config.json` unless overridden by CLI flags.
- `label-targets` and `review` are marked manual/GUI and skipped in automated timing.
