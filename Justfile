set shell := ["bash", "-cu"]

default:
    @just --list

venv:
    uv venv .venv

build: venv
    uv sync --all-packages
    for pkg in packages/*; do \
      if [[ -f "$pkg/pyproject.toml" ]]; then \
        uv build "$pkg"; \
      fi; \
    done

clean:
    find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".ruff_cache" -o -name ".mypy_cache" \) -prune -exec rm -rf {} +
    rm -rf dist build .coverage htmlcov
    for pkg in packages/*; do \
      rm -rf "$pkg/dist" "$pkg/build"; \
      find "$pkg" -type d -name "*.egg-info" -prune -exec rm -rf {} +; \
    done

fclean: clean
    rm -rf .venv

fetch-dataset: build
    uv run dataset-fetcher
