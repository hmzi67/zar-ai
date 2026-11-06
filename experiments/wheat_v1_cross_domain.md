# wheat_v1 cross-domain evaluation

Scores the frozen wheat vision v1 model (EfficientNetV2-S, four seeds) on external datasets it never saw. The tables below are generated from MLflow by `python -m zarai.vision.eval_external`, and contain only measured numbers. [TBD] means not run.

To see what's missing and where to get it, run `.venv/bin/python scripts/check_external_datasets.py`.

<!-- results:start -->
Frozen model runs: `e4123b0516144fb687e9cf942ca6c6f7`, `94142b122b784279a811f3930a058d13`, `1ec14ad01b6046f28aca8625798a00f6`, `1daa65830603479bbed9cc5c67c9f30d`.

Results are the mean ± SD over the 4 seeds logged in the MLflow experiment `wheat_v1_cross_domain`.

## Dataset status

| dataset | status | images by class | skipped |
| --- | --- | --- | --- |
| mendeley_wheat_leaf_ethiopia | ready | healthy 102, septoria 97, yellow_rust 208 | - |
| mendeley_iari_leaf_rust | ready | brown_rust 368, healthy 491 | - |
| nwrd | ready | yellow_rust 99 | mask size differs: 1 |
| cgiar_wheat_rust | ready | black_rust 376, brown_rust 357, healthy 142 | outside train/: 610 |

## Results

Macro-F1 averages the dataset's mapped classes only, and a prediction of any of our 15 classes counts. "Outside" is the share of predictions in classes the dataset doesn't contain. NWRD has one mapped class, so its macro-F1 depends only on recall.

| evaluation | images | macro-F1 | accuracy | outside | seen in train or val | macro-F1 without those |
| --- | --- | --- | --- | --- | --- | --- |
| mendeley_wheat_leaf_ethiopia | 407 | 0.3249 ± 0.0045 | 0.2488 ± 0.0012 | 0.7402 ± 0.0061 | 182 | 0.2903 ± 0.0185 |
| mendeley_iari_leaf_rust | 859 | 0.6459 ± 0.1433 | 0.5573 ± 0.1552 | 0.4159 ± 0.1503 | 31 | 0.6468 ± 0.1432 |
| nwrd_image | 99 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0 | 0.0000 ± 0.0000 |
| nwrd_tiles | 2109 | 0.0278 ± 0.0083 | 0.0141 ± 0.0043 | 0.9859 ± 0.0043 | 0 | 0.0278 ± 0.0083 |
| cgiar_wheat_rust | 875 | 0.6261 ± 0.0117 | 0.5857 ± 0.0094 | 0.2857 ± 0.0090 | 420 | 0.4079 ± 0.0113 |

## With abstention at the val-chosen threshold

| evaluation | coverage | macro-F1 on answered | accuracy on answered | outside |
| --- | --- | --- | --- | --- |
| mendeley_wheat_leaf_ethiopia | 0.8913 ± 0.0270 | 0.3321 ± 0.0008 | 0.2752 ± 0.0112 | 0.7227 ± 0.0114 |
| mendeley_iari_leaf_rust | 0.2561 ± 0.1294 | 0.6871 ± 0.1605 | 0.7041 ± 0.2112 | 0.2777 ± 0.2158 |
| nwrd_image | 0.4672 ± 0.0788 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| nwrd_tiles | 0.4388 ± 0.0159 | 0.0027 ± 0.0033 | 0.0014 ± 0.0016 | 0.9986 ± 0.0016 |
| cgiar_wheat_rust | 0.6371 ± 0.0163 | 0.7411 ± 0.0210 | 0.7635 ± 0.0177 | 0.1402 ± 0.0117 |

## Per-class recall

With abstention, an image the model declines counts as a miss.

| evaluation | class | images | recall | coverage | recall with abstention |
| --- | --- | --- | --- | --- | --- |
| mendeley_wheat_leaf_ethiopia | healthy | 102 | 0.9926 ± 0.0049 | 0.9779 ± 0.0147 | 0.9779 ± 0.0147 |
| mendeley_wheat_leaf_ethiopia | septoria | 97 | 0.0000 ± 0.0000 | 0.6340 ± 0.1278 | 0.0000 ± 0.0000 |
| mendeley_wheat_leaf_ethiopia | yellow_rust | 208 | 0.0000 ± 0.0000 | 0.9688 ± 0.0224 | 0.0000 ± 0.0000 |
| mendeley_iari_leaf_rust | brown_rust | 368 | 0.3356 ± 0.2183 | 0.1957 ± 0.1812 | 0.0421 ± 0.0368 |
| mendeley_iari_leaf_rust | healthy | 491 | 0.7235 ± 0.1143 | 0.3014 ± 0.1845 | 0.2856 ± 0.1925 |
| nwrd_image | yellow_rust | 99 | 0.0000 ± 0.0000 | 0.4672 ± 0.0788 | 0.0000 ± 0.0000 |
| nwrd_tiles | yellow_rust | 2109 | 0.0141 ± 0.0043 | 0.4388 ± 0.0159 | 0.0006 ± 0.0007 |
| cgiar_wheat_rust | black_rust | 376 | 0.9249 ± 0.0085 | 0.8670 ± 0.0075 | 0.8438 ± 0.0059 |
| cgiar_wheat_rust | brown_rust | 357 | 0.2479 ± 0.0340 | 0.4328 ± 0.0333 | 0.1190 ± 0.0238 |
| cgiar_wheat_rust | healthy | 142 | 0.5370 ± 0.0202 | 0.5423 ± 0.0223 | 0.4630 ± 0.0133 |

## Confusion

Mean counts over the seeds, for true labels by predicted.

### mendeley_wheat_leaf_ethiopia

| true \ predicted | blast | brown_rust | common_root_rot | healthy | leaf_blight | tan_spot |
| --- | --- | --- | --- | --- | --- | --- |
| healthy | 0.0 | 0.8 | 0.0 | 101.2 | 0.0 | 0.0 |
| septoria | 1.0 | 86.2 | 3.5 | 3.2 | 0.8 | 2.2 |
| yellow_rust | 0.0 | 206.5 | 0.0 | 1.2 | 0.2 | 0.0 |

### mendeley_iari_leaf_rust

| true \ predicted | aphid | black_rust | blast | brown_rust | healthy | leaf_blight | mildew | mite | septoria | smut | tan_spot | yellow_rust |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| brown_rust | 0.0 | 4.2 | 0.0 | 123.5 | 8.5 | 11.8 | 58.2 | 0.0 | 0.2 | 0.0 | 158.5 | 3.0 |
| healthy | 2.8 | 0.0 | 1.5 | 14.5 | 355.2 | 0.2 | 67.2 | 1.2 | 0.0 | 0.2 | 46.8 | 1.2 |

### nwrd_image

| true \ predicted | aphid | black_rust | blast | brown_rust | common_root_rot | healthy | leaf_blight | mildew | septoria | stem_fly | tan_spot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yellow_rust | 0.8 | 6.2 | 0.5 | 0.8 | 1.0 | 74.8 | 1.0 | 6.5 | 7.0 | 0.2 | 0.2 |

### nwrd_tiles

| true \ predicted | aphid | black_rust | blast | brown_rust | common_root_rot | healthy | leaf_blight | mildew | mite | septoria | smut | stem_fly | tan_spot | yellow_rust |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yellow_rust | 18.0 | 1.2 | 1.2 | 1065.0 | 29.2 | 692.8 | 41.0 | 115.8 | 8.2 | 46.0 | 0.5 | 9.8 | 50.5 | 29.8 |

### cgiar_wheat_rust

| true \ predicted | aphid | black_rust | blast | brown_rust | common_root_rot | fusarium_head_blight | healthy | leaf_blight | mildew | mite | septoria | smut | stem_fly | tan_spot | yellow_rust |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| black_rust | 1.0 | 347.8 | 3.8 | 6.2 | 2.8 | 0.5 | 0.8 | 3.5 | 1.8 | 0.0 | 2.0 | 0.0 | 0.0 | 5.5 | 0.5 |
| brown_rust | 3.0 | 75.8 | 1.0 | 88.5 | 1.2 | 2.0 | 26.5 | 83.5 | 8.5 | 1.2 | 14.0 | 0.5 | 0.0 | 43.0 | 8.2 |
| healthy | 6.2 | 3.0 | 13.0 | 0.2 | 4.2 | 1.5 | 76.2 | 8.8 | 0.5 | 0.5 | 0.0 | 22.8 | 1.2 | 3.8 | 0.0 |

Per-seed matrices are in each run's `confusion/` artifacts.

## Overlap with wheat_v1

External images within pHash distance 6 of a wheat_v1 image, by their label in each dataset. Only overlaps with train or val are left out of "macro-F1 without those". pHash also matches different leaves on plain black backgrounds, so check these pairs by eye.

| evaluation | external label | wheat_v1 label | wheat_v1 split | images |
| --- | --- | --- | --- | --- |
| mendeley_wheat_leaf_ethiopia | yellow_rust | brown_rust | train | 83 |
| mendeley_wheat_leaf_ethiopia | healthy | healthy | train | 72 |
| mendeley_wheat_leaf_ethiopia | yellow_rust | brown_rust | test | 16 |
| mendeley_wheat_leaf_ethiopia | healthy | healthy | val | 15 |
| mendeley_wheat_leaf_ethiopia | healthy | healthy | test | 14 |
| mendeley_wheat_leaf_ethiopia | yellow_rust | brown_rust | val | 12 |
| mendeley_iari_leaf_rust | healthy | mildew | train | 25 |
| mendeley_iari_leaf_rust | brown_rust | mildew | train | 6 |
| cgiar_wheat_rust | black_rust | black_rust | train | 246 |
| cgiar_wheat_rust | black_rust | black_rust | test | 58 |
| cgiar_wheat_rust | black_rust | black_rust | val | 57 |
| cgiar_wheat_rust | healthy | healthy | train | 54 |
| cgiar_wheat_rust | brown_rust | black_rust | train | 20 |
| cgiar_wheat_rust | brown_rust | brown_rust | train | 16 |
| cgiar_wheat_rust | healthy | healthy | test | 10 |
| cgiar_wheat_rust | healthy | healthy | val | 8 |
| cgiar_wheat_rust | brown_rust | leaf_blight | train | 5 |
| cgiar_wheat_rust | brown_rust | leaf_blight | val | 4 |
| cgiar_wheat_rust | brown_rust | brown_rust | val | 3 |
| cgiar_wheat_rust | brown_rust | tan_spot | train | 3 |
| cgiar_wheat_rust | brown_rust | black_rust | test | 2 |
| cgiar_wheat_rust | brown_rust | black_rust | val | 2 |
| cgiar_wheat_rust | black_rust | brown_rust | val | 1 |
| cgiar_wheat_rust | brown_rust | brown_rust | test | 1 |
| cgiar_wheat_rust | brown_rust | mildew | test | 1 |
| cgiar_wheat_rust | brown_rust | mildew | val | 1 |
| cgiar_wheat_rust | brown_rust | tan_spot | test | 1 |
<!-- results:end -->

## Interpretation

All figures are from the tables above.

### Two of the four datasets aren't truly unseen

- **Ethiopia:** 212 of 407 images are already in `wheat_v1`, including 101 of its 102 healthy images. 111 of its 208 stripe rust images are in `wheat_v1` labelled **`brown_rust`**, in train, val and test. Spot checks show the same photos (pHash distance 0 or 2), and in the original Kaggle dataset they sit in the `Brown Rust` folder.
- **CGIAR:** 493 of 875 images are in `wheat_v1`, including 361 of its 376 stem rust images. Some CGIAR leaf rust images carry other labels in `wheat_v1`: `black_rust` (24), `leaf_blight` (9), `tan_spot` (4) and `mildew` (2).
- **IARI:** the 31 matches are false positives. Checked by eye, they are different leaves that pHash matches only because both sit on plain black backgrounds. So IARI is effectively unseen, and its "seen" count shouldn't be taken at face value.
- **NWRD:** no overlap.

For Ethiopia and CGIAR, "macro-F1 without those" is the better measure, but the remaining images still come from the same sources the model trained on.

### Results on unseen data are poor

- **NWRD (stripe rust in field canopy photos):** no whole image is predicted `yellow_rust`. Most are predicted `healthy` (74.8 of 99 on average). On 384 px tiles that contain rust, `yellow_rust` recall is 0.0141 ± 0.0043, and half the tiles are predicted `brown_rust` (1,065 of 2,109).
- **Ethiopia:** stripe rust is never predicted `yellow_rust` (0 of 208); almost all of it is predicted `brown_rust`. Septoria, which isn't in `wheat_v1`, is also mostly predicted `brown_rust` (86.2 of 97). Only healthy, which the model saw in training, is recognised (0.9926).
- **IARI (single leaves on a masked black background):** macro-F1 is 0.6459 ± 0.1433, the least stable across seeds. Leaf rust recall is 0.3356 ± 0.2183; it's more often predicted `tan_spot` or `mildew`.
- **CGIAR:** stem rust recall is high (0.9249), but most of those images were in training. Leaf rust recall is 0.2479, and macro-F1 without the overlapping images is 0.4079 ± 0.0113.

### Abstention doesn't catch these errors

The val-chosen threshold still answers 97% of Ethiopia stripe rust images (coverage 0.9688), which are all wrong. It answers 47% of NWRD images (0.4672), also all wrong. The model is confidently wrong on stripe rust outside `wheat_v1`, so abstention can't stand in for coverage of the domain.

### Likely causes

These are hypotheses that fit the measurements but aren't proven here.

- **Stripe rust:** across both external stripe rust datasets, the model hardly ever predicts `yellow_rust`. Its `yellow_rust` class may come from a narrow set of sources, while at least 111 stripe rust images in training are labelled `brown_rust`. That would teach the model that this appearance means `brown_rust`.
- **Sources:** Ethiopian septoria predicted as `brown_rust` fits a source shortcut. The only Ethiopian-farm diseased images in training are the stripe rust ones labelled `brown_rust`.
- **Image type:** NWRD shows whole canopies and IARI masked single leaves. Both differ from the mostly close-up `wheat_v1` photos.

### Implications

- **Wheat vision v1 shouldn't be used for stripe rust or leaf rust outside `wheat_v1`.** It isn't field-ready for any class tested here except possibly stem rust, and that result is mostly memorised.
- **Before any `wheat_v2`:** the `brown_rust` and `yellow_rust` labels need review by someone with disease expertise, starting with the 111 matched images. External datasets should be deduplicated against `wheat_v1` when the dataset is built, by checking pixels as well as pHash because of the black-background false matches.
- **Training data:** the external datasets would be more useful as training data split by source than as test sets, with field photos kept back as the real test.
