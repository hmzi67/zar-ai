# wheat_v1 leave-source-out

Tests whether EfficientNetV2-S learned disease features or this source's appearance. Models train and select on `wheat_v1` without the 923 watermarked images, then score those images as an unseen source. The tables are generated from the MLflow experiment `wheat_v1_leave_source_out` by `bash scripts/run_wheat_v1_leave_source_out.sh`.

<!-- results:start -->
## Runs

| seed | MLflow run ID | val macro-F1 | best epoch / epochs | test macro-F1 |
| --- | --- | --- | --- | --- |
| 42 | `9f299562188e4de8a9ab4a435265f4ce` | 0.8335 | 24 / 25 | 0.8152 |
| 0 | `af6d20f40b204c608db4530d18c1ceb3` | 0.8397 | 16 / 21 | 0.8091 |
| 1 | `3a1aceb0b2ed4558a6051b2f7cdd461f` | 0.8371 | 18 / 23 | 0.8131 |
| 2 | `1215f894fe4a44b7b375bfaa3d0ab4a1` | 0.8318 | 15 / 20 | 0.8157 |

## Held-out source: recall per class

All 923 watermarked images, which these models never saw in training or selection.

| class | images | recall | true-class probability |
| --- | --- | --- | --- |
| mildew | 461 | 0.3265 ± 0.0502 | 0.2264 ± 0.0418 |
| fusarium_head_blight | 462 | 0.7944 ± 0.0336 | 0.5239 ± 0.0355 |

Accuracy over all 923: 0.5607 ± 0.0199.

## Test scores compared with the baseline

The baseline trained with the watermarked images. Both columns use the same test images.

| measure | baseline | trained without the source |
| --- | --- | --- |
| Test macro-F1, all images | 0.8469 ± 0.0028 | 0.8133 ± 0.0030 |
| Test accuracy, all images | 0.8681 ± 0.0034 | 0.8305 ± 0.0034 |
| Test macro-F1, without watermarked images | 0.8273 ± 0.0024 | 0.8197 ± 0.0048 |
| `mildew` recall, watermarked test images | 1.0000 ± 0.0000 | 0.2763 ± 0.0483 |
| `mildew` recall, clean test images | 0.8841 ± 0.0155 | 0.8795 ± 0.0114 |
| `fusarium_head_blight` recall, watermarked test images | 1.0000 ± 0.0000 | 0.8250 ± 0.0214 |
| `fusarium_head_blight` recall, clean test images | 0.6979 ± 0.0399 | 0.6771 ± 0.0524 |

## Breakdown of the held-out source

| subset | images | recall |
| --- | --- | --- |
| `mildew`, black background | 411 | 0.3479 ± 0.0509 |
| `mildew`, other background | 50 | 0.1500 ± 0.0577 |
| `mildew`, from original train split | - | 0.3283 ± 0.0507 |
| `mildew`, from original val split | - | 0.3576 ± 0.0560 |
| `mildew`, from original test split | - | 0.2763 ± 0.0483 |
| `fusarium_head_blight`, from original train split | - | 0.7876 ± 0.0300 |
| `fusarium_head_blight`, from original val split | - | 0.7945 ± 0.0680 |
| `fusarium_head_blight`, from original test split | - | 0.8250 ± 0.0214 |

Coverage at the val-chosen threshold: 0.3613 ± 0.0191; accuracy when answering: 0.8779 ± 0.0413.

## What held-out images were predicted as

Counts summed over the seeds.

`mildew`: mildew 602, aphid 363, yellow_rust 332, leaf_blight 231, tan_spot 148, mite 67.

`fusarium_head_blight`: fusarium_head_blight 1468, blast 104, yellow_rust 100, aphid 73, smut 35, mite 32.

<!-- results:end -->

## Interpretation

### What this shows

**The baseline's near-perfect recall on the watermarked images doesn't carry over to an unseen source.** On those images, recall falls from 1.000 to 0.276 for `mildew` and from 1.000 to 0.825 for `fusarium_head_blight` (test images, same images in both columns). Over all 923 images, it is 0.327 and 0.794.

- **`mildew`:** models trained only on the other mildew images miss most of this source, mainly predicting `aphid`, `yellow_rust` and `leaf_blight`. The baseline's result on this source depends on having trained on it.
- **`fusarium_head_blight`:** most of the recall survives, so these models do transfer disease or head appearance to this source. Recall on clean fusarium images (0.677) is lower than on this source, so these images look easier than the clean ones. Those clean figures rest on 24 test images.
- **Clean images:** training with the source adds little for them. Macro-F1 without the watermarked images is 0.8273 for the baseline and 0.8197 here, and clean `mildew` recall is 0.884 and 0.880.
- **Test macro-F1:** on all test images, this model scores 0.8133 against the baseline's 0.8469. That 0.034 gap comes almost entirely from the source's images, so the baseline's test score benefits from the source being in both train and test.

### What it doesn't show

- **Not proof that the model reads the watermark:** the watermark check found little reliance on the text. These results fit learning the source's overall look, such as close-up leaves on black backgrounds, but this experiment can't identify which cue matters.
- **Not a pure shortcut measure:** this source differs visually from the rest of `mildew`, and models here never saw a black background. Some of the drop is an ordinary shift in appearance. That's also what any new source would present, which is why it matters for field use.
- **Not a field estimate:** this is one unseen source, for two classes only. The other 13 classes have no equivalent test here, and one source can't show how they'd behave.

### Implications

- Treat `mildew` recall on a new source as unknown, and possibly far below the test figure. Unseen-source results (0.33 and 0.79) should be reported beside the 0.8469 test macro-F1.
- Macro-F1 without the watermarked images (about 0.82-0.83) is a better summary of the rest of the data than 0.8469, but it still isn't a field estimate.
- Splitting by source would give a more honest test than the current split, but only this one source can be identified so far. The cross-domain evaluation tests other sources directly.
