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
        rm -rf "$pkg/dist" "$pkg/build"; \
        mkdir -p "$pkg/dist"; \
        uv build "$pkg" --out-dir "$pkg/dist"; \
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
    @rm -rf dataset/{{dataset}}
    @uv run dataset-fetcher --dataset {{dataset}}

label-targets: venv
    @uv sync --all-packages
    @rm -rf dataset/targets/images dataset/targets/labels
    @uv run target-labeller

generate-dataset: venv
    @uv sync --all-packages
    @rm -rf dataset/augmented/{{dataset}}
    @uv run dataset-generator --dataset {{dataset}}

check-dataset: venv
    @uv sync --all-packages
    @rm -rf dataset/augmented/{{dataset}}/reports
    @uv run augment-checker --dataset {{dataset}}

train: venv
    @uv sync --all-packages
    @rm -rf artifacts/detector-train/runs/current artifacts/detector-train/eval_predictions/current artifacts/detector-train/eval_reports/current
    @uv run detector-train --dataset {{dataset}}

eval: venv
    @uv sync --all-packages
    @rm -rf predictions dataset/augmented/{{dataset}}/grade_reports
    @latest_json="artifacts/detector-train/latest_run.json"; \
    weights=$(uv run python -c "import json,sys; print(json.loads(open(sys.argv[1], encoding='utf-8').read())['weights_best'])" "$latest_json"); \
    model_key=$(uv run python -c "import json,sys; from pathlib import Path; from detector_grader.data import infer_model_name_from_weights; payload=json.loads(open(sys.argv[1], encoding='utf-8').read()); print(infer_model_name_from_weights(Path(payload['weights_best'])))" "$latest_json"); \
    uv run detector-infer --dataset {{dataset}} --weights "$weights" --model-name "$model_key"; \
    uv run detector-grader --dataset {{dataset}} --model "$model_key" --no-run-inference

review: venv
    @uv sync --all-packages
    @uv run detector-reviewer --dataset {{dataset}} --model latest
