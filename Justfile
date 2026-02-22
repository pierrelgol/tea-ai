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

generate-augmented-robust OUT_ROOT='dataset/augmented_v2/{{dataset}}': venv
    @uv sync --all-packages
    @uv run dataset-generator \
      --dataset {{dataset}} \
      --dataset-root dataset \
      --output-root {{OUT_ROOT}} \
      --complexity-profile obb_robust_v1 \
      --targets-per-image-min 2 \
      --targets-per-image-max 4 \
      --max-occlusion-ratio 0.45 \
      --allow-partial-visibility \
      --blur-prob 0.35 \
      --motion-blur-prob 0.20 \
      --noise-prob 0.25 \
      --jpeg-artifact-prob 0.20 \
      --color-jitter-prob 0.50

check-augmented: venv
    @uv sync --all-packages
    @uv run augment-checker --dataset {{dataset}} --datasets-base-root dataset/augmented

check-augmented-root DATASET_ROOT='dataset/augmented_v2/{{dataset}}': venv
    @uv sync --all-packages
    @uv run augment-checker --dataset-root {{DATASET_ROOT}} --no-gui

evaluate-detector: venv
    @uv sync --all-packages
    @uv run detector-evaluator --dataset {{dataset}} --datasets-base-root dataset/augmented --predictions-root predictions

train-detector: venv
    @uv sync --all-packages
    @uv run detector-train --dataset {{dataset}} --datasets-base-root dataset/augmented --artifacts-root artifacts/detector-train

train-detector-root DATASET_ROOT='dataset/augmented_v2/{{dataset}}': venv
    @uv sync --all-packages
    @uv run detector-train --dataset-root {{DATASET_ROOT}} --artifacts-root artifacts/detector-train --wandb --wandb-mode online

optimize-detector: venv
    @uv sync --all-packages
    @uv run detector-train-optimize --dataset {{dataset}} --datasets-base-root dataset/augmented --artifacts-root artifacts/detector-train --predictions-root predictions --baseline-file baseline.txt

infer-detector WEIGHTS MODEL: venv
    @uv sync --all-packages
    @uv run detector-infer --weights {{WEIGHTS}} --model-name {{MODEL}} --dataset {{dataset}} --datasets-base-root dataset/augmented --output-root predictions

review-detector: venv
    @uv sync --all-packages
    @uv run detector-reviewer --dataset {{dataset}} --datasets-base-root dataset/augmented

grade-detector MODEL='latest': venv
    @uv sync --all-packages
    @uv run detector-grader --dataset {{dataset}} --datasets-base-root dataset/augmented --predictions-root predictions --model {{MODEL}}

grade-detector-root DATASET_ROOT='dataset/augmented_v2/{{dataset}}' MODEL='latest': venv
    @uv sync --all-packages
    @uv run detector-grader --dataset-root {{DATASET_ROOT}} --predictions-root predictions --artifacts-root artifacts/detector-train --model {{MODEL}}

quality-gate GRADE_REPORT DATASET_ROOT='dataset/augmented_v2/{{dataset}}' MIN_RUN_GRADE='': venv
    @uv sync --all-packages
    @if [[ -n "{{MIN_RUN_GRADE}}" ]]; then \
      uv run dataset-generator-gate \
        --integrity-report {{DATASET_ROOT}}/reports/integrity_report.json \
        --geometry-report {{DATASET_ROOT}}/reports/geometry_report.json \
        --grade-report {{GRADE_REPORT}} \
        --max-geometry-outlier-rate 0.005 \
        --min-run-grade {{MIN_RUN_GRADE}}; \
    else \
      uv run dataset-generator-gate \
        --integrity-report {{DATASET_ROOT}}/reports/integrity_report.json \
        --geometry-report {{DATASET_ROOT}}/reports/geometry_report.json \
        --grade-report {{GRADE_REPORT}} \
        --max-geometry-outlier-rate 0.005; \
    fi
