# Public datasets

**Nothing downloads automatically.** Before downloading, open the source page, confirm the licence allows research use and redistribution of derived models, and record it. Licence fields below are for you to fill in from the source page.

| Dataset | Link | Licence | Checked by / date |
|---|---|---|---|
| PlantVillage | https://github.com/spMohanty/PlantVillage-Dataset | [CHECK] | |
| PlantDoc | https://github.com/pratikkayal/PlantDoc-Dataset | [CHECK] | |
| Cotton leaf disease (Mendeley Data) | Search: https://data.mendeley.com/research-data/?search=cotton%20leaf%20disease | [CHECK] | |
| Rice leaf disease (Mendeley Data) | Search: https://data.mendeley.com/research-data/?search=rice%20leaf%20disease | [CHECK] | |

Record the exact Mendeley dataset DOI and version once chosen.

## After downloading

1. Unpack into `data/raw/images_public/<dataset>/`.
2. Build metadata with `site_id` starting `public_` (see `data/README.md`).
3. Run `dvc add data/raw/images_public`, then commit the `.dvc` file.
