# Image annotation guideline (template)

## Class list
To fill in with the plant pathology expert.

| Crop | Label | Description | Key symptoms | Look-alikes |
|---|---|---|---|---|
| [CROP] | [LABEL] | | | |

## Photo protocol
- Three views per plant: `whole_plant`, `leaf_closeup`, `leaf_underside`.
- At least 3 phone models across the collection.
- Natural light; no flash.
- One affected plant in focus. No faces, hands with identifying marks, or names.

## Metadata
Required for every image, per `src/zarai/data/schemas/image_meta.schema.json`: `image_id`, `file_path`, `crop`, `label`, `site_id`, `district`, `date`, `growth_stage`, `view`, `phone_model`, `labeller`, `expert_verified`, `unsure`.

## The "unsure" bin
If you cannot decide on a label, set `unsure: true` and do not guess. An expert reviews these images. They stay out of all splits until resolved.

## Quality check
A double-labelled subset is scored with `python -m zarai.data.agreement`. Agreement target: [KAPPA].
