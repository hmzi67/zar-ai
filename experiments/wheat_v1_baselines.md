# wheat_v1 baselines

Three pretrained baselines on `data/processed/wheat_v1`, with splits from `data/splits/wheat_v1.csv`. Each run is logged to MLflow in `./mlruns`, in the experiment `wheat_v1_baselines`.

The results below are generated from logged runs by `python -m zarai.vision.results`, which writes [TBD] for anything not logged. The model is selected on validation macro-F1 only. Test results never choose a model or tune anything.

<!-- results:start -->
Dataset DVC hashes: processed `a806332e8b58e98b6af893139bd3a17c.dir`, splits `a301478ca1c041f875119eac6039b76b`, raw `cd676a8f5dba83c342d8607559dbea5e.dir`.

## Baselines (seed 42)

| model | val macro-F1 (best) | best epoch / epochs run | test macro-F1 (95% CI) | test accuracy (95% CI) | test ECE before | test ECE after | temperature | MLflow run ID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EfficientNetV2-S | 0.8534 | 12 / 17 | 0.8489 (0.8294-0.8667) | 0.8677 (0.8506-0.8834) | 0.1164 | 0.0217 | 0.706 | `e4123b0516144fb687e9cf942ca6c6f7` |
| ConvNeXt-T | 0.8577 | 6 / 11 | 0.8432 (0.8228-0.8618) | 0.8658 (0.8494-0.8822) | 0.0655 | 0.0367 | 0.822 | `d18401f6f1b542daa8754c868c818fb2` |
| DINOv2 ViT-S/14 linear probe | 0.8079 | 17 / 22 | 0.7990 (0.7779-0.8185) | 0.8210 (0.8028-0.8399) | 0.1273 | 0.0214 | 0.646 | `384466b4f41f41979fb414cfef7726f7` |

Highest seed 42 validation macro-F1: **ConvNeXt-T**.

## Reruns of ConvNeXt-T

| seed | val macro-F1 (best) | best epoch / epochs run | test macro-F1 (95% CI) | test accuracy (95% CI) | test ECE before | test ECE after | temperature | MLflow run ID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed 0 | 0.8646 | 12 / 17 | 0.8384 (0.8178-0.8574) | 0.8626 (0.8462-0.8796) | 0.0708 | 0.0570 | 0.894 | `463f5baedfd34f5aac8c6470160ec47b` |
| seed 1 | 0.8634 | 5 / 10 | 0.8362 (0.8158-0.8562) | 0.8570 (0.8393-0.8752) | 0.0475 | 0.0342 | 0.841 | `73189d0b546244a2b54e49774c3396e2` |
| seed 2 | 0.8447 | 2 / 7 | 0.8205 (0.7984-0.8399) | 0.8507 (0.8318-0.8683) | 0.0644 | 0.0236 | 0.785 | `089fa23997524d8b89f902dd1db33d53` |

## All logged seeds

| run | val macro-F1 (best) | best epoch / epochs run | test macro-F1 (95% CI) | test accuracy (95% CI) | test ECE before | test ECE after | temperature | MLflow run ID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EfficientNetV2-S, seed 0 | 0.8521 | 22 / 25 | 0.8463 (0.8260-0.8636) | 0.8689 (0.8526-0.8847) | 0.0996 | 0.0276 | 0.720 | `94142b122b784279a811f3930a058d13` |
| EfficientNetV2-S, seed 1 | 0.8521 | 21 / 25 | 0.8431 (0.8228-0.8621) | 0.8639 (0.8469-0.8809) | 0.1028 | 0.0233 | 0.709 | `1ec14ad01b6046f28aca8625798a00f6` |
| EfficientNetV2-S, seed 2 | 0.8602 | 23 / 25 | 0.8491 (0.8278-0.8668) | 0.8721 (0.8557-0.8878) | 0.1066 | 0.0181 | 0.705 | `1daa65830603479bbed9cc5c67c9f30d` |
| EfficientNetV2-S, seed 42 | 0.8534 | 12 / 17 | 0.8489 (0.8294-0.8667) | 0.8677 (0.8506-0.8834) | 0.1164 | 0.0217 | 0.706 | `e4123b0516144fb687e9cf942ca6c6f7` |
| ConvNeXt-T, seed 0 | 0.8646 | 12 / 17 | 0.8384 (0.8178-0.8574) | 0.8626 (0.8462-0.8796) | 0.0708 | 0.0570 | 0.894 | `463f5baedfd34f5aac8c6470160ec47b` |
| ConvNeXt-T, seed 1 | 0.8634 | 5 / 10 | 0.8362 (0.8158-0.8562) | 0.8570 (0.8393-0.8752) | 0.0475 | 0.0342 | 0.841 | `73189d0b546244a2b54e49774c3396e2` |
| ConvNeXt-T, seed 2 | 0.8447 | 2 / 7 | 0.8205 (0.7984-0.8399) | 0.8507 (0.8318-0.8683) | 0.0644 | 0.0236 | 0.785 | `089fa23997524d8b89f902dd1db33d53` |
| ConvNeXt-T, seed 42 | 0.8577 | 6 / 11 | 0.8432 (0.8228-0.8618) | 0.8658 (0.8494-0.8822) | 0.0655 | 0.0367 | 0.822 | `d18401f6f1b542daa8754c868c818fb2` |
| DINOv2 ViT-S/14 linear probe, seed 42 | 0.8079 | 17 / 22 | 0.7990 (0.7779-0.8185) | 0.8210 (0.8028-0.8399) | 0.1273 | 0.0214 | 0.646 | `384466b4f41f41979fb414cfef7726f7` |

## Mean and SD across seeds

| model | seeds | val macro-F1 | test macro-F1 | test accuracy | test ECE after | accuracy when answering |
| --- | --- | --- | --- | --- | --- | --- |
| EfficientNetV2-S | 0, 1, 2, 42 | 0.8544 ± 0.0039 | 0.8469 ± 0.0028 | 0.8681 ± 0.0034 | 0.0227 ± 0.0039 | 0.9489 ± 0.0053 |
| ConvNeXt-T | 0, 1, 2, 42 | 0.8576 ± 0.0091 | 0.8346 ± 0.0098 | 0.8590 ± 0.0067 | 0.0379 ± 0.0140 | 0.9479 ± 0.0072 |

## Confusion between similar classes (test)

Each cell is the count of `a` predicted as `b`, then `b` predicted as `a`.

| pair | EfficientNetV2-S | ConvNeXt-T | DINOv2 ViT-S/14 linear probe |
| --- | --- | --- | --- |
| `black_rust` / `brown_rust` | 3 / 2 | 2 / 3 | 1 / 7 |
| `leaf_blight` / `septoria` | 0 / 0 | 0 / 0 | 0 / 2 |
| `leaf_blight` / `tan_spot` | 9 / 12 | 7 / 17 | 9 / 27 |

## Background and watermark checks (test)

Grad-CAM grids are in each run's `plots/gradcam_grid.png`. If the mass on dark pixels is well above the dark area, the model is looking at the background. Watermarks are only checked by viewing the grid.

| check | EfficientNetV2-S | ConvNeXt-T | DINOv2 ViT-S/14 linear probe |
| --- | --- | --- | --- |
| Mildew recall, black background | 1.000 | 1.000 | 1.000 |
| Mildew recall, other | 0.874 | 0.916 | 0.815 |
| Mildew: Grad-CAM mass on dark pixels | 0.177 | 0.108 | 0.194 |
| Mildew: dark area | 0.166 | 0.166 | 0.166 |
| Fusarium: Grad-CAM mass on dark pixels | 0.013 | 0.011 | 0.024 |
| Fusarium: dark area | 0.012 | 0.012 | 0.012 |

## Abstention (test, threshold chosen on val)

| model | threshold | coverage | accuracy on answered |
| --- | --- | --- | --- |
| EfficientNetV2-S | 0.771 | 0.7914 | 0.9435 |
| ConvNeXt-T | 0.874 | 0.7895 | 0.9513 |
| DINOv2 ViT-S/14 linear probe | 0.612 | 0.7958 | 0.9177 |

## ONNX export of the selected model

Largest absolute logit difference from PyTorch: 1.34e-05 (required: at most 1e-4).

## Per-class results for the selected model (test)

| class | precision | recall | F1 | support |
| --- | --- | --- | --- | --- |
| aphid | 0.9259 | 0.7692 | 0.8403 | 130 |
| black_rust | 0.7308 | 0.8444 | 0.7835 | 45 |
| blast | 0.8172 | 0.8736 | 0.8444 | 87 |
| brown_rust | 0.9598 | 0.9278 | 0.9435 | 180 |
| common_root_rot | 0.8395 | 0.8095 | 0.8242 | 84 |
| fusarium_head_blight | 0.9053 | 0.9149 | 0.9101 | 94 |
| healthy | 0.9042 | 0.9679 | 0.9350 | 156 |
| leaf_blight | 0.6250 | 0.7317 | 0.6742 | 82 |
| mildew | 0.9023 | 0.9401 | 0.9208 | 167 |
| mite | 0.8033 | 0.8596 | 0.8305 | 114 |
| septoria | 0.9623 | 0.9808 | 0.9714 | 52 |
| smut | 0.7746 | 0.7333 | 0.7534 | 75 |
| stem_fly | 0.9200 | 0.8846 | 0.9020 | 26 |
| tan_spot | 0.6232 | 0.4674 | 0.5342 | 92 |
| yellow_rust | 0.9710 | 0.9901 | 0.9805 | 203 |

## Most common errors for the selected model (test)

| true | predicted | count |
| --- | --- | --- |
| tan_spot | leaf_blight | 17 |
| aphid | mite | 9 |
| mite | tan_spot | 8 |
| leaf_blight | tan_spot | 7 |
| smut | blast | 7 |
| tan_spot | mite | 7 |
| aphid | leaf_blight | 5 |
| aphid | mildew | 5 |
| blast | fusarium_head_blight | 5 |
| smut | leaf_blight | 5 |
<!-- results:end -->

## Error analysis

Based on the nine logged runs above. Seed statistics cover seeds 0, 1, 2 and 42, with sample standard deviations. Counts that aren't in the tables come from each run's logged `predictions/test_logits.npz` and `metrics/test_errors.json`.

### Summary

ConvNeXt-T and EfficientNetV2-S are tied on the primary metric, mean val macro-F1. EfficientNetV2-S is the working choice, on val-based stability and calibration (see "Model comparison"). Field performance is unmeasured. `wheat_v1_watermark_check.md` measured the watermark directly and found little reliance on the text itself, but whether one source inflates these scores is still open.

**Decision:** EfficientNetV2-S is frozen as wheat vision v1 (`configs/vision/wheat_vision_v1.yaml`). Its best-val seed is registered as the `candidate`, which doesn't clear it for field use: `wheat_v1_leave_source_out.md` shows that test scores overstate `mildew` on an unseen source.

### Model comparison

- **Primary metric:** mean val macro-F1 is 0.8576 ± 0.0091 for ConvNeXt-T and 0.8544 ± 0.0039 for EfficientNetV2-S. The difference (0.0031) is well inside seed noise, so this is a tie.
- **Tie-breakers from val only:** EfficientNetV2-S varies less between seeds, its last-five-epoch mean val macro-F1 is about the same or slightly higher (0.8501 vs 0.8462), and its val ECE after scaling is lower (0.0218 vs 0.0362). ConvNeXt-T has the lower minimum val loss (0.4944 vs 0.5257).
- **Test, not used to choose:** EfficientNetV2-S has macro-F1 0.8469 ± 0.0028 against 0.8346 ± 0.0098, and is higher on each paired seed. Its val-to-test gap is also smaller (0.0076 vs 0.0230), which fits ConvNeXt-T's best val scores being lucky peaks. This has now been seen, so it can't serve as a clean test of the choice.
- **Fairness limit:** three of the four ConvNeXt-T runs stopped with the learning rate still high, while the three EfficientNetV2-S reruns ran all 25 epochs. Its 3e-4 learning rate may suit ConvNeXt-T less, so this compares these settings, not the architectures.

### Watermarks and backgrounds

- **EfficientNetV2-S Grad-CAM:** the first grids used a layer before BatchNorm and were meaningless. They were regenerated from the same checkpoints, with no change to any other metric. The corrected maps (seeds 42 and 2 viewed) still show grid-like heat near the image edges, including on black corners of mildew images and along the top of `fusarium_head_blight` images, where the watermark often starts. They aren't reliable enough to say what this model uses, so its reliance on watermarks is unresolved.
- **Measured reliance on watermarks:** see `wheat_v1_watermark_check.md`. Removing the text changes macro-F1 by at most 0.0033, and painting it on clean images moves few predictions, so the Grad-CAM impressions below overstate the text's role. 411 of the 415 black-background mildew images also carry the text, so these are largely one source rather than two separate cues.
- **Watermarks in Grad-CAM:** the ConvNeXt-T Grad-CAM grids for seeds 42 and 0 show activation following the watermark text on several `fusarium_head_blight` images, at about 0.90 confidence or more, and on some black-background `mildew` images. This is a visual check of 64 images across the two grids, not a measurement of how common it is.
- **Black backgrounds:** this isn't proven as a shortcut. Mildew recall is 48/48 on black backgrounds in all nine runs (exact 95% lower bound 0.926), against 0.874-0.933 on other mildew images for those two models. But none of the 9 black-background test images from other classes was predicted as mildew in any of the nine runs. Nine images is too few to rule it out, and 48 of the 57 black-background test images are mildew.
- **Limit of the metric:** Grad-CAM mass on dark pixels doesn't detect this problem, because the watermark text isn't dark. The dark-area figures also vary from 0.108 to 0.350 between seeds, because each seed samples different images, so they can't be compared across seeds.

### Confusions

- **`tan_spot`:** this is the weakest class, with a mean F1 of 0.521 (range 0.457-0.580) and mean recall of 0.473. 32 of its 92 test images are wrong in all four seeds.
- **`tan_spot` and `leaf_blight`:** the errors are asymmetric: `tan_spot` predicted as `leaf_blight` happens 16-25 times per seed, and the reverse 5-8 times. `leaf_blight` has a mean F1 of 0.666.
- **Possible data problems:** the 14 test images misclassified between this pair in all four seeds include a rice disease collage, a captioned figure, side-by-side composites and distant canopy shots. Some of this error is likely to come from the data, but the labels need review by someone who can identify these diseases.
- **`black_rust` / `brown_rust` and `leaf_blight` / `septoria`:** these have few direct confusions (at most 4 and 2 per direction per seed). The images with conflicting labels for these pairs were excluded from the dataset, so this doesn't show the model can separate these diseases on hard cases.
- **Consistent errors:** 107 of 1,587 test images are wrong in all four seeds, and 374 are wrong in at least one.

### Training stability

- **ConvNeXt-T early stopping:** validation macro-F1 moves by up to about 0.035 between epochs from epoch 3 on, so patience 5 stops training early. Seed 2 chose epoch 2 and stopped at epoch 7 with the learning rate still at 2.56e-4, and it has the lowest test macro-F1 (0.8205). No ConvNeXt-T seed reached the low-learning-rate end of the cosine schedule.
- **EfficientNetV2-S:** its seed 42 run stopped at epoch 17, and seeds 0, 1 and 2 ran all 25 epochs with best epochs 21-23. Val macro-F1 is smoother, and its seeds differ much less.

### Calibration and abstention

- **Temperature:** it is below 1 in all four seeds (0.785-0.894), meaning the raw model is underconfident, which is expected with label smoothing.
- **ECE:** scaling lowers test ECE in every seed, to 0.0236-0.0570. In the seed 42 reliability diagram, confidence still exceeds accuracy in most bins between about 0.3 and 0.9.
- **Abstention:** the threshold from val answers 77.9-79.0% of test images at 94.2-95.6% accuracy. Abstention can't catch a confident shortcut, so these figures have the same watermark limitation.
