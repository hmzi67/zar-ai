# imagenette2_160: source

Non-plant photos, used only as negatives to calibrate and test the chatbot's plant filter
(`zarai.vision.plant_filter`). Never used to train or select a disease model.

| Item | Value |
| --- | --- |
| Page | <https://github.com/fastai/imagenette> |
| Licence | ImageNet terms (a subset of ImageNet; research use) |
| Downloaded from | https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz |
| Archive | `imagenette2-160.tgz`, 99,003,388 bytes |
| Archive sha256 | `64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5` |
| Archive md5 | `e793b78cc4c9e9a4ccc0c1155377a412` |
| Checked against | no published checksum; recorded at download on 2026-11-12 |
| Unpacked layout | {train,val}/<wordnet id>/*.JPEG, 160 px short side, 10 classes |
| DVC | `imagenette2_160.dvc` (the unpacked folder) |

The 10 classes are tench, English springer, cassette player, chain saw, church, French horn,
garbage truck, gas pump, golf ball and parachute. Several are outdoors on grass or with
vegetation behind them (springer, golf ball, tench, church), which makes them harder negatives.
The archive was deleted after unpacking, and these checksums were recorded first.
