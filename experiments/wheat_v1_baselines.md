# wheat_v1 baselines

Three pretrained baselines on `data/processed/wheat_v1`, with splits from `data/splits/wheat_v1.csv`. Each run is logged to MLflow in `./mlruns`, in the experiment `wheat_v1_baselines`.

The results below are generated from logged runs by `python -m zarai.vision.results`, which writes [TBD] for anything not logged. The model is selected on validation macro-F1 only. Test results never choose a model or tune anything.

<!-- results:start -->
Dataset DVC hashes: processed `a806332e8b58e98b6af893139bd3a17c.dir`, splits `a301478ca1c041f875119eac6039b76b`, raw `cd676a8f5dba83c342d8607559dbea5e.dir`.

## Baselines (seed 42)

| model | val macro-F1 (best) | best epoch / epochs run | test macro-F1 (95% CI) | test accuracy (95% CI) | test ECE before | test ECE after | temperature | MLflow run ID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EfficientNetV2-S | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| ConvNeXt-T | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| DINOv2 ViT-S/14 linear probe | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

Selected on validation macro-F1: **[TBD]**.

## Reruns of the selected model

| seed | val macro-F1 (best) | best epoch / epochs run | test macro-F1 (95% CI) | test accuracy (95% CI) | test ECE before | test ECE after | temperature | MLflow run ID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed 0 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| seed 1 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| seed 2 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

## Confusion between similar classes (test)

Each cell is the count of `a` predicted as `b`, then `b` predicted as `a`.

| pair | EfficientNetV2-S | ConvNeXt-T | DINOv2 ViT-S/14 linear probe |
| --- | --- | --- | --- |
| `black_rust` / `brown_rust` | [TBD] | [TBD] | [TBD] |
| `leaf_blight` / `septoria` | [TBD] | [TBD] | [TBD] |
| `leaf_blight` / `tan_spot` | [TBD] | [TBD] | [TBD] |

## Background and watermark checks (test)

Grad-CAM grids are in each run's `plots/gradcam_grid.png`. If the mass on dark pixels is well above the dark area, the model is looking at the background. Watermarks are only checked by viewing the grid.

| check | EfficientNetV2-S | ConvNeXt-T | DINOv2 ViT-S/14 linear probe |
| --- | --- | --- | --- |
| Mildew recall, black background | [TBD] | [TBD] | [TBD] |
| Mildew recall, other | [TBD] | [TBD] | [TBD] |
| Mildew: Grad-CAM mass on dark pixels | [TBD] | [TBD] | [TBD] |
| Mildew: dark area | [TBD] | [TBD] | [TBD] |
| Fusarium: Grad-CAM mass on dark pixels | [TBD] | [TBD] | [TBD] |
| Fusarium: dark area | [TBD] | [TBD] | [TBD] |

## Abstention (test, threshold chosen on val)

| model | threshold | coverage | accuracy on answered |
| --- | --- | --- | --- |
| EfficientNetV2-S | [TBD] | [TBD] | [TBD] |
| ConvNeXt-T | [TBD] | [TBD] | [TBD] |
| DINOv2 ViT-S/14 linear probe | [TBD] | [TBD] | [TBD] |

## ONNX export of the selected model

Largest absolute logit difference from PyTorch: [TBD] (required: at most 1e-4).

## Per-class results for the selected model (test)

[TBD]

## Most common errors for the selected model (test)

[TBD]
<!-- results:end -->

## Error analysis

[TBD]: write this after reviewing the logged runs, confusion matrices and Grad-CAM grids.
