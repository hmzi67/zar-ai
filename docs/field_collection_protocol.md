# Field collection protocol

## Before visiting
1. Get permission from the landowner or farmer at each site, and record it. Use `docs/consent_form_urdu_template.md`.
2. Assign a `site_id` that does not identify anyone, such as `site_01`.
3. Check phone storage, battery and the date setting.

## During collection
Follow the photo protocol in `docs/annotation_guideline_images.md`.

## Metadata
Record `site_id`, `district`, `date`, `growth_stage`, `view` and `phone_model` for each image.

## Privacy
- No faces, names, phone numbers or house fronts in photos.
- No GPS coordinates or personal details in the metadata. Remove EXIF location before upload.
- Keep signed consent forms separate from the data, stored securely by the project lead.

## Upload
1. Copy photos to `data/raw/images_field/<site_id>/<date>/`.
2. Add metadata rows and run `python -m zarai.data.validate --schema image_meta <file>`.
3. Run `dvc add data/raw/images_field`, commit the `.dvc` file, then run `dvc push`.
4. Delete the photos from personal phones after confirming the push.
