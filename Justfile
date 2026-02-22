set shell := ["bash", "-cu"]
dataset := "coco128"

default:
    @just --list

venv:
    @if [[ ! -d .venv ]]; then uv venv .venv; fi

build: venv
    @uv sync --all-packages
    @shopt -s nullglob
    @for pkg in packages/*; do \
      if [[ -f "$pkg/pyproject.toml" ]]; then \
        if compgen -G "$pkg/dist/*.whl" > /dev/null && compgen -G "$pkg/dist/*.tar.gz" > /dev/null; then \
          :; \
        else \
          mkdir -p "$pkg/dist"; \
          uv build "$pkg" --out-dir "$pkg/dist"; \
        fi; \
      fi; \
    done

clean:
    @find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".ruff_cache" -o -name ".mypy_cache" \) -prune -exec rm -rf {} +
    @for path in artifacts dataset predictions wandb; do \
      if [[ -e "$path" ]]; then rm -rf "$path"; fi; \
    done
    @if [[ -d dist || -d build || -f .coverage || -d htmlcov ]]; then \
      rm -rf dist build .coverage htmlcov; \
    else \
      :; \
    fi
    @shopt -s nullglob
    @for pkg in packages/*; do \
      rm -rf "$pkg/dist" "$pkg/build"; \
      find "$pkg" -type d -name "*.egg-info" -prune -exec rm -rf {} +; \
    done

fclean: clean
    @if [[ -d .venv ]]; then rm -rf .venv; fi
    @if [[ -d .pixi ]]; then rm -rf .pixi; fi
    @if [[ -d .tox ]]; then rm -rf .tox; fi
    @if [[ -d .nox ]]; then rm -rf .nox; fi
    @if [[ -d .hypothesis ]]; then rm -rf .hypothesis; fi
    @find . -maxdepth 1 -type f -name "*.pt" -delete

fetch-dataset: venv
    @uv sync --all-packages
    @uv run dataset-fetcher --dataset {{dataset}}

label-targets: venv
    @uv sync --all-packages
    @uv run target-labeller

generate-dataset: venv
    @uv sync --all-packages
    @uv run dataset-generator --dataset {{dataset}}

check-dataset: venv
    @uv sync --all-packages
    @uv run augment-checker --dataset {{dataset}}

train: venv
    @uv sync --all-packages
    @uv run detector-train --dataset {{dataset}}

eval: venv
    @uv sync --all-packages
    @weights=$$(uv run python - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("artifacts/detector-train/latest_run.json").read_text(encoding="utf-8"))
print(payload["weights_best"])
PY
); \
    model_key=$$(uv run python - <<'PY'
from pathlib import Path
from detector_grader.data import infer_model_name_from_weights
import json

payload = json.loads(Path("artifacts/detector-train/latest_run.json").read_text(encoding="utf-8"))
print(infer_model_name_from_weights(Path(payload["weights_best"])))
PY
); \
    uv run detector-infer --dataset {{dataset}} --weights "$$weights" --model-name "$$model_key"; \
    uv run detector-grader --dataset {{dataset}} --model "$$model_key" --run-inference false

review MODEL='latest': venv
    @uv sync --all-packages
    @uv run detector-reviewer --dataset {{dataset}} --model {{MODEL}}
