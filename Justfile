set shell := ["bash", "-cu"]

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
    @if [[ ! -d dataset/coco8 ]] || [[ -z "$(ls -A dataset/coco8 2>/dev/null)" ]]; then \
      uv run dataset-fetcher --dataset-root dataset; \
    fi

label-targets: venv
    @uv sync --all-packages
    @uv run target-labeller --images-dir targets --labels-dir dataset/targets/labels --classes-file dataset/targets/classes.txt

generate-augmented: venv
    @uv sync --all-packages
    @uv run dataset-generator --output-root dataset/augmented
