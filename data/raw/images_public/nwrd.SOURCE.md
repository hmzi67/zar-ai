# nwrd: source

| Item | Value |
| --- | --- |
| Page | <https://zenodo.org/records/21672928> |
| Licence | CC BY 4.0 |
| Downloaded from | https://zenodo.org/api/records/21672928/files/NWRD.zip/content |
| Archive | `NWRD.zip`, 608,898,312 bytes |
| Archive sha256 | `7493a66318fb16cff0a45cc47361b0fbf34d38bfe8a15351b575c21063ab9385` |
| Archive md5 | `7b9193efebe1ed8c60b10c9f05d11e14` |
| Checked against | the source's published checksum, on 2026-11-06 |
| Unpacked layout | {train,test}/images/<id>.jpg with {train,test}/masks/<id>.png |
| DVC | `nwrd.dvc` (the unpacked folder) |

The archive was deleted after unpacking, and these checksums were recorded first. `scripts/check_external_datasets.py --download nwrd` fetches it again, and checks it against the same hash.

Note: Images and tiles with an empty mask are skipped, not treated as healthy.
