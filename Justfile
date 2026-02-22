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

check-augmented-root DATASET_ROOT='dataset/augmented/{{dataset}}': venv
    @uv sync --all-packages
    @uv run augment-checker --dataset-root {{DATASET_ROOT}} --no-gui

evaluate-detector: venv
    @uv sync --all-packages
    @uv run detector-evaluator --dataset {{dataset}} --datasets-base-root dataset/augmented --predictions-root predictions

train-detector: venv
    @uv sync --all-packages
    @uv run detector-train --dataset {{dataset}} --datasets-base-root dataset/augmented --artifacts-root artifacts/detector-train

train-detector-root DATASET_ROOT='dataset/augmented/{{dataset}}': venv
    @uv sync --all-packages
    @uv run detector-train --dataset-root {{DATASET_ROOT}} --artifacts-root artifacts/detector-train

infer-detector WEIGHTS MODEL: venv
    @uv sync --all-packages
    @uv run detector-infer --weights {{WEIGHTS}} --model-name {{MODEL}} --dataset {{dataset}} --datasets-base-root dataset/augmented --output-root predictions

review-detector: venv
    @uv sync --all-packages
    @uv run detector-reviewer --dataset {{dataset}} --datasets-base-root dataset/augmented

grade-detector MODEL='latest': venv
    @uv sync --all-packages
    @uv run detector-grader --dataset {{dataset}} --datasets-base-root dataset/augmented --predictions-root predictions --model {{MODEL}}

grade-detector-root DATASET_ROOT='dataset/augmented/{{dataset}}' MODEL='latest': venv
    @uv sync --all-packages
    @uv run detector-grader --dataset-root {{DATASET_ROOT}} --predictions-root predictions --artifacts-root artifacts/detector-train --model {{MODEL}}
