# detector-train

> Lean detector training pipeline with DINOv3 feature distillation.

`detector-train` is a specialized training harness built on top of the Ultralytics framework. It implements a multi-stage distillation process where a DINOv3 teacher model guides a YOLOv8-OBB student, forcing the student to learn high-dimensional semantic features while maintaining real-time inference speeds.

## Architecture

The trainer orchestrates a complex dance between the YOLO student and a frozen DINOv3 teacher. Feature maps are extracted from both models, aligned via interpolation, and compared using a masked cosine-distance loss.

```text
       [ DINOv3 Teacher ]             [ YOLO Student ]
         (Frozen, ViT)                 (Trainable, CNN)
               |                              |
               v                              v
      +-----------------+            +-----------------+
      | Feature Map (T) |            | Feature Map (S) |
      +--------+--------+            +--------+--------+
               |                              |
               +-------------->+<-------------+
                               |
                   [ DISTILLATION ENGINE ]
            1. Spatial Alignment (Interpolation)
            2. OBB-Masked Cosine Distance Loss
            3. Dynamic Weight Scaling (Warmup/Stages)
                               |
                               v
                   [ COMBINED LOSS FUNCTION ]
               L_total = L_yolo + w_distill * L_distill
```

## Technical Core

### 1. DINOv3 Feature Distillation
Unlike standard training, `detector-train` injects a custom loss term:
*   **Layer Hooking**: Automatically attaches forward hooks to specific student layers (e.g., Backbone/Neck) to capture intermediate feature maps.
*   **Spatial Masking**: Uses ground-truth Oriented Bounding Boxes (OBB) to generate binary masks. The distillation loss can be weighted differently for "Object" pixels vs "Background" pixels.
*   **Cosine Similarity Alignment**: Encourages the student's feature vectors to point in the same high-dimensional direction as the DINOv3 teacher's semantic features.

### 2. Multi-Stage Training Protocol
The training is split into two distinct phases to maximize stability:
*   **Stage A (Frozen Backbone)**: The student's backbone layers are frozen. The model focuses entirely on aligning its detection heads and neck with the teacher's semantic space.
*   **Stage B (Full Finetuning)**: The backbone is unfrozen, allowing the entire network to adapt to the specific dataset while still being regularized by the distillation loss.

### 3. Monitoring & Logging
*   **W&B Integration**: Automatic logging of YOLO loss components plus internal distillation metrics (`dino_loss`, `obj_distill_loss`, `bg_distill_loss`).
*   **DINO Visualization**: Periodically exports "signal maps" showing exactly where the student is struggling to match the teacher's semantic representation.
*   **W&B Fallback**: Automatically switches to offline mode if the API key is missing or the network is unreachable.

## Usage

```bash
uv run detector-train --config config.json
```

The trainer automatically:
1.  Generates a `data.yaml` following the `PipelineLayout` paths.
2.  Resolves local DINOv3 weights from `artifacts/models/dino/`.
3.  Initializes the `DinoOBBTrainer` with the provided distillation configuration.
4.  Executes the Stage A/B curriculum.
