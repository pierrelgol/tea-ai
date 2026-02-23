# dataset-generator

> Procedural synthesis of Oriented Bounding Box (OBB) datasets via homographic projection.

The `dataset-generator` package is the core data engine of the fine-tuning pipeline. It generates high-fidelity synthetic training data by mathematically projecting canonical "target" images onto real-world "background" images. This avoids the immense human cost of manually labeling oriented objects across thousands of variations in scale, perspective, and occlusion.

## Architecture

The generator operates on a curriculum-driven loop. It pulls canonical targets, samples randomized homographies, checks for valid projection geometries, handles occlusions, applies photometric augmentations, and exports the final composited image alongside precise YOLO OBB labels.

```text
  [ Canonical Targets ]       [ Backgrounds ]       [ Grade Reports ]
    (Images + Polys)            (Train/Val)         (Curriculum Data)
           |                         |                      |
           v                         v                      v
+-----------------------------------------------------------------------+
|                           DATASET GENERATOR                           |
|                                                                       |
|  1. CURRICULUM RESOLVER: Adjusts sampling biases based on history.     |
|  2. HOMOGRAPHY SAMPLER: 3x3 projection with convexity/area checks.     |
|  3. SYNTHESIS ENGINE: Alpha-blending + Poisson (optional) compositing. |
|  4. PHOTOMETRIC STACK: Multi-stage jitter, blur, and noise.           |
|  5. METADATA EXPORTER: Detailed H-matrix and state logging.           |
+------------------------------------+----------------------------------+
                                     |
                                     v
                        [ Synthetic OBB Dataset ]
                        (images/, labels/, meta/)
```

## Technical Core

### 1. Homographic Projection
The package uses $3\times3$ homography matrices ($H$) to warp 2D canonical targets into realistic 3D perspective representations. Every sample is validated against strict geometric constraints:
*   **Convexity**: The projected quad must remain a convex polygon.
*   **Boundary Enforcement**: All four corners must remain within the background image frame.
*   **Minimum Area**: Targets below a specific pixel-area threshold (based on $H$ scale) are rejected to prevent training on sub-pixel noise.
*   **Aspect Ratio Sanity**: Prevents degenerate projections that result in extreme "slivers".

### 2. Photometric Augmentation Stack
To bridge the "Sim-to-Real" gap, a randomized stack is applied post-synthesis:
*   **HSV Jitter**: Randomized hue shifts and saturation/value gains.
*   **Blur Suite**: Gaussian blur and directional motion blur (simulating camera movement).
*   **Noise Models**: Additive Gaussian noise and JPEG compression artifacting.
*   **Alpha Blending**: Smooth transition between target edges and background textures.

### 3. Curriculum-Driven Synthesis
The generator is "aware" of model performance. By parsing `grade_reports` from previous runs:
*   **Hard Class Mining**: Increases the sampling frequency of classes with low MAP.
*   **Difficulty Scaling**: If the model masters simple perspectives, the generator increases `perspective_jitter` and decreases the `min_quad_area_frac` to force the model to learn smaller, more distorted targets.

## Usage

```python
from pathlib import Path
from dataset_generator.config import GeneratorConfig
from dataset_generator.generator import generate_dataset

# 1. Define synthesis constraints
config = GeneratorConfig(
    output_root=Path("./dataset/augmented"),
    target_images_dir=Path("./dataset/targets/images"),
    target_labels_dir=Path("./dataset/targets/labels"),
    # ... other config params
    curriculum_enabled=True,
)

# 2. Execute procedural generation
results = generate_dataset(config)
```

## Metadata Schema
Every generated image has a corresponding `.json` in the `meta/` directory containing:
*   `H`: The $3\times3$ homography matrix used for projection.
*   `class_name`: The canonical label.
*   `photometric_flags`: Boolean flags for which augmentations were applied.
*   `visible_ratio`: The percentage of the target not occluded by other objects.
