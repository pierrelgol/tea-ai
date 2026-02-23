# detector-infer

> High-throughput inference engine for Oriented Bounding Box (OBB) models.

`detector-infer` is a focused execution module designed to run trained YOLO-OBB models across large datasets. It prioritizes precision and strict adherence to the OBB format, ensuring that downstream grading and review tools receive standardized geometric data.

## Architecture

The inference engine abstracts the complexity of hardware acceleration and batching, providing a clean path from raw weights to normalized YOLO-OBB text files.

```text
    [ Weights (.pt) ]          [ Image Split ]
           |                         |
           v                         v
+-------------------------------------------------------+
|                   DETECTOR INFER                      |
|                                                       |
|  1. MODEL LOADER: Loads YOLO OBB with strict validation. |
|  2. BATCH PROCESSOR: Efficient GPU/MPS utilization.      |
|  3. COORDINATE NORMALIZER: Maps px to [0, 1] space.      |
|  4. OBB ENFORCER: Fails if model outputs HBB only.       |
+---------------------------+---------------------------+
                            |
                            v
                [ Prediction Labels ]
           (Normalized YOLO OBB + Confidence)
```

## Technical Core

### 1. Strict OBB Enforcement
Unlike general-purpose inference scripts, `detector-infer` explicitly validates the model output structure. If the loaded weights produce standard Horizontal Bounding Boxes (HBB) instead of the required 8-point OBB format, the pipeline will terminate with a `RuntimeError`. This ensures data integrity throughout the finetuning loop.

### 2. Coordinate Normalization
Predictions are automatically mapped from the internal inference resolution (`imgsz`) back to the original image dimensions. Coordinates are then normalized to a `[0, 1]` range:
*   **Format**: `class_id x1 y1 x2 y2 x3 y3 x4 y4 confidence`
*   **Precision**: High-precision floats are used for coordinates to prevent rounding errors during geometric grading.

### 3. Batching & Hardware Abstraction
The module utilizes `pipeline-runtime-utils` to automatically detect the best available hardware (CUDA, MPS, or CPU). Batch sizes are configurable via the global `config.json` to optimize throughput for specific GPU VRAM constraints.

## Usage

```bash
uv run detector-infer --config config.json
```

The engine will:
1.  Locate the best weights for the current `model_key`.
2.  Iterate through the `train` and `val` splits of the configured dataset.
3.  Write result files to `artifacts/<model_key>/runs/<run_id>/labels/<split>/`.
