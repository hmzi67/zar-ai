# mendeley_wheat_leaf_ethiopia: source

| Item | Value |
| --- | --- |
| Page | <https://data.mendeley.com/datasets/wgd66f8n6h/1> |
| Licence | CC BY 4.0 |
| Downloaded from | https://data.mendeley.com/public-files/datasets/wgd66f8n6h/files/4ec9266d-d7c1-4fd5-85c9-e5624c09a847/file_downloaded |
| Archive | `wheat_leaf.rar`, 1,511,923,897 bytes |
| Archive sha256 | `23653dd88f9f900315a9a0c88a163c891b3bdd00cc89b36d98cc5b3fcc1d7e94` |
| Archive md5 | `e2b72fdc3ce321df9ecd48cf7b24b72a` |
| Checked against | the source's published checksum, on 2026-11-06 |
| Unpacked layout | <any depth>/{healthy,stripe_rust,septoria}/*.jpg (a train/val/test level is fine) |
| DVC | `mendeley_wheat_leaf_ethiopia.dvc` (the unpacked folder) |

The archive was deleted after unpacking, and these checksums were recorded first. `scripts/check_external_datasets.py --download mendeley_wheat_leaf_ethiopia` fetches it again, and checks it against the same hash.

Note: A .rar archive: it is unpacked if unrar is installed, otherwise use 7z x wheat_leaf.rar.
