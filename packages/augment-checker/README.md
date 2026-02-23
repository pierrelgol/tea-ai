# augment-checker

> Specialized QA and integrity verification suite for augmented datasets.

`augment-checker` is a multi-modal verification tool designed to ensure the quality and consistency of procedural OBB datasets. It combines automated filesystem audits, geometric sanity checks, and a high-performance visual browser to catch "Sim-to-Real" failures before they reach the training stage.

## Architecture

The checker operates as a three-tier system: the Auditor (checks files), the Analyst (checks geometry), and the Browser (visual verification).

```text
    [ Augmented Dataset ] --------+
             |                    |
             v                    v
+-------------------------+  +-------------------------+
|      DATA AUDITOR       |  |     GEOMETRY ANALYST    |
| (Filesystem & YOLO)     |  | (Metadata & H-Matrix)   |
+------------+------------+  +------------+------------+
             |                            |
             +-------------+--------------+
                           |
                           v
+-------------------------+-------------------------+
|                    VISUAL BROWSER                 |
|       (PySide6 GUI + Hardware Acceleration)       |
|                                                   |
| - Fast Sample Navigation                          |
| - Ground Truth & OBB Metadata Overlays           |
| - Prediction Comparison (Optional)               |
+---------------------------------------------------+
```

## Technical Core

### 1. Multi-Level Integrity Auditing
*   **Filesystem Sync**: Verifies 1:1 mapping between `images/`, `labels/`, and `meta/` directories.
*   **YOLO Protocol**: Validates OBB label format (4 corner pairs) and ensures coordinates are normalized [0, 1].
*   **Class Mapping**: Ensures all synthesized samples belong to the current project's class registry.

### 2. Geometric Sanity Metrics
By parsing the `meta/` JSON files produced by `dataset-generator`, the checker evaluates:
*   **H-Matrix Integrity**: Checks for singular or degenerate homographies.
*   **Projection Validations**: Recalculates quad area and convexity to detect sub-pixel artifacts or edge-case clipping errors.
*   **Visible Ratio**: Identifies targets that are too heavily occluded by other objects in the synthesis stack.

### 3. PySide6 Visual Browser
A high-performance desktop application for manual QA:
*   **Hardware Accelerated Overlays**: Renders complex OBB quads and class labels directly on high-resolution images.
*   **Filter Engine**: Allows developers to filter for specific splits or samples with identified integrity issues.
*   **Debug Exports**: Generates hard-burned visual overlays for sharing or documentation.

## Usage

```bash
uv run augment-checker --config config.json
```

The tool will:
1.  Perform a full scan of the augmented dataset.
2.  Emit a summary report to the console.
3.  Launch the GUI for deep inspection of samples and their associated metadata.
