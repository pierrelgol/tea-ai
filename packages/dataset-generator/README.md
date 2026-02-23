# dataset-generator

Lean synthetic dataset generator.

## Run

```bash
uv run dataset-generator --config config.json
```

## Data contract

- Backgrounds: `dataset/<dataset>/images/train|val`
- Targets: `dataset/targets/images`, `dataset/targets/labels`, `dataset/targets/classes.txt`

## Output

- `dataset/augmented/<dataset>/images/train|val`
- `dataset/augmented/<dataset>/labels/train|val`
- `dataset/augmented/<dataset>/meta/train|val`
- `dataset/augmented/<dataset>/classes.txt`
- `dataset/augmented/<dataset>/classes_map.json`
- `dataset/augmented/<dataset>/split_audit.json`
