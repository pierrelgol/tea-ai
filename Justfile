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

fetch-dataset: venv
    @uv sync --all-packages
    @uv run dataset-fetcher --dataset {{dataset}} --dataset-root dataset

label-targets DATASET_ROOT='dataset' TARGETS_SUBDIR='targets': venv
    @uv sync --all-packages
    @uv run target-labeller --dataset-root {{DATASET_ROOT}} --targets-subdir {{TARGETS_SUBDIR}} --images-dir targets

generate-augmented: venv
    @uv sync --all-packages
    @uv run dataset-generator --dataset {{dataset}} --dataset-root dataset

check-augmented: venv
    @uv sync --all-packages
    @uv run augment-checker --dataset {{dataset}} --datasets-base-root dataset/augmented

evaluate-detector: venv
    @uv sync --all-packages
    @uv run detector-evaluator --dataset {{dataset}} --datasets-base-root dataset/augmented --predictions-root predictions

train-detector: venv
    @uv sync --all-packages
    @uv run detector-train --dataset {{dataset}} --datasets-base-root dataset/augmented --artifacts-root artifacts/detector-train

infer-detector WEIGHTS MODEL: venv
    @uv sync --all-packages
    @uv run detector-infer --weights {{WEIGHTS}} --model-name {{MODEL}} --dataset {{dataset}} --datasets-base-root dataset/augmented --output-root predictions

review-detector: venv
    @uv sync --all-packages
    @uv run detector-reviewer --dataset {{dataset}} --datasets-base-root dataset/augmented
