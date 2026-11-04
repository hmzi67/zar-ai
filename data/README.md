# Data

Contents are tracked with DVC and gitignored. **Never commit raw data.** Only `.gitkeep`, `README.md` and `*.dvc` files go into git.

| Folder | What goes here | Owner |
|---|---|---|
| `raw/kb_sources/` | Original KB documents | RA-3 |
| `raw/qa/` | Q&A pairs (JSONL) | RA-3 |
| `raw/images_public/` | Public image datasets, after a licence check | RA-2 |
| `raw/images_field/` | Field photos | RA-2 |
| `interim/` | Intermediate outputs, can be regenerated | Whoever runs the step |
| `processed/` | Final model inputs, can be regenerated | Whoever runs the step |
| `splits/` | Frozen train/val/test splits | Hamza |

## Naming

- KB source: `<source_id>.<ext>`, with a matching row in `kb/source_register.csv`.
- Q&A id: `qa_0001` (pattern `^qa_\d{4,}$`).
- Field image: `<site_id>/<date>/<image_id>.jpg`, with `date` as `YYYY-MM-DD`.
- Public dataset `site_id`: starts with `public_`, for example `public_plantdoc_<group>`.

## Site IDs and splits

`split_images.py` keeps each `site_id` in one split. Sites starting with `public_` never enter `field_test`. A public dataset with a single `site_id` moves whole into train or val, so use finer groups where the dataset provides them.

`unsure` images are excluded from all splits.

## Metadata columns

`image_meta.schema.json` requires every field, which suits field images. Public images have no real `date`, `view` or `phone_model`, so decide with RA-2 whether to relax the schema or keep separate metadata for them before validating.

## Tracking with DVC

```bash
dvc add data/raw/images_field
git add data/raw/images_field.dvc
dvc push   # after: dvc remote add -d storage <remote-url>
```
