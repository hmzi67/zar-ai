# data/processed/qa — wheat Q&A v1

> **⚠ Synthetic data.** Both source files are template-generated. Provenance is unverified, every
> record has `source_type = llm` and `expert_verified = false`. **Do not use this data as a
> test/evaluation set or as knowledge-base ground truth.** It is suitable only as (weak) training data
> after expert review of the flagged records.

Built by `dvc repro wheat_qa_v1` (`zarai.data.wheat_qa`, config `configs/data/wheat_qa_v1.yaml`)
from `data/qa/ZarAI_Wheat_Classification.xlsx` (file A) and `data/qa/wheat only qa dataset..xlsx`
(file B). The source scan is in `data/qa/SCAN_REPORT.md`.

| File | What |
|---|---|
| `wheat_qa_v1.jsonl` | 579 records = 193 unique Q&As × 3 languages (`ur`, `ur-Latn`, `en`). Schema `qa_record`. |
| `wheat_qa_variants.jsonl` | 35 778 records = 11 926 file-B rows × 3 languages: greeting/city/acreage wrappers and extra paraphrases of the 178 v1 file-B groups. **Train-only augmentation.** |
| `safety_refusals.jsonl` | 612 records: the 3 file-B "safe refusal" groups (e.g. "tell me the pesticide dose" → refuse, refer to the agriculture office), base rows and their variants. Not crop knowledge, so not in v1. |
| `KB_CANDIDATES.md` | Checklist of the 35 tier-1 v1 rows (source cites agripunjab), to match one by one against the original agripunjab page. |
| `wheat_qa_v1.summary.json` | All counts, including where every input row ended up (`row_accounting`) and counts per `kb_tier`. |
| `../../interim/qa_review.csv` | 145 quarantined rows for expert review (not in v1, variants or safety). |
| `../../interim/qa_cleaning_log.csv` | One line per normalised cell, dropped/quarantined row and moved paraphrase. |

## Why the data is synthetic

- File B is 223 base Q&As × 68 template variants (`Assalam o alaikum. … My field is in Multan. Kindly guide me.`).
- 400 of file A's 539 questions are `(Case N)` templates that share a few generic answers; the
  `source_type` labels on identical rows differ (PARC_guidelines / NARC_field_study / …), so they
  look randomly assigned. One generic answer (soil preparation, NPK, timely sowing, certified
  seed) is given to "best agronomic practices for *X*" questions of six different topics.

## Rules applied

- **Normalisation** (`zarai.nlp.normalise.normalise_text`): NFKC, Arabic ي/ى/ك/ه/ة → Urdu ی/ک/ہ,
  zero-width/bidi marks and vowel diacritics removed (madda and hamza kept), Urdu-script cells get
  `؟ ، ؛ ۔`, one space between words. 764 cells changed; mostly diacritics. Answer wording is never
  changed — the only edit to questions is removing the `(Case N)` / `(کیس N)` suffix.
- **PII**: phone numbers, CNICs and e-mails would become `[PHONE]`/`[CNIC]`/`[EMAIL]`; none were found.
  The city names and acreages in file B's wrappers are synthetic and kept.
- **Quarantine → `qa_review.csv`** (column `reason`). Every row carrying a quarantined answer is
  excluded, in both files, including the 67 variants of each affected file-B group. The regex
  rules are `quarantine_rules` in the config, checked in order; the first match names the reason:
  - `unsafe_answer`: sprayed aluminium phosphide + methyl bromide, tricyclazole for rust, 2,4-D for wild oats.
  - `scientifically_wrong`: seed treatment said to prevent yellow/leaf/stem rust or powdery mildew
    (airborne, not seed-borne). Seed treatment for Karnal bunt, Fusarium, Septoria and tan spot stays.
  - `template_fill_wrong`: one fill-in answer pasted onto every subject — one treatment for all
    diseases (spray fungicide / "متاثرہ پتوں کو ہٹائیں", incl. mosaic, a virus), one control for all
    pests (insecticide / yellow sticky traps for stem borer, caterpillar, mite, beetles), one symptom
    for all nutrient deficiencies ("پودا سست ہو جاتا ہے" / "پتے پیلے پڑ جاتے ہیں"), one control for
    all weeds (herbicide / "ہاتھ سے نکالیں"), and the irrigation template ("in summer", "at <stage>
    every 7-10 / 15 days"). Both wordings of each fill-in are caught, in file A and in file B's
    `existing_dataset` rows. "wheat needs 3-4 irrigations in Winter" is not a fill-in error and stays.
  - `verify_current`: the answer names specific varieties (NARC-2009, Faisalabad-2008, Pakistan-2013,
    NARC-2019, Faisalabad-2020, …); check against the current approved list before use.
  - `conflicting_answers`: one question, two genuinely different answers. Two wordings of one answer
    (rapidfuzz ratio ≥ 85, e.g. "Quick nutrition in heat or water stress" / "Provides quick
    nutrition in heat and water stress") are not a conflict: the file-B copy is kept. After the rules
    above, no conflicts remain.
  - `question_answer_mismatch`: an answer the files give to questions of ≥ 2 crop topics
  (a generic template answer) used for a question whose topic it never mentions — the
  "best agronomic practices" answer for weed management, pest management, post harvest and
  organic farming (4 questions, 132 file-A rows; one review line per question, `source_rows` = how
  many rows it stands for). The same answer stays in v1 where it fits (variety selection, planting
  calendar, nutrient deficiency, fertilizer management, soil health). The check runs before
  duplicate removal so a fitting question is never folded into a mismatched one. Keyword absence
  alone is not used: specific answers often skip the topic word ("how to recognise rust?" →
  "yellow stripes on the leaf").
- **One per group**: v1 keeps the first `original` row of each remaining file-B group (174 groups).
  The group's other originals and wrapped variants go to `wheat_qa_variants.jsonl`.
- **Duplicates**: same English question + answer (after normalisation and `(Case N)` removal), or
  same answer with question similarity ≥ 0.90 (rapidfuzz ratio), is dropped and logged. File B wins
  over file A. Rows with the same question but different answers are conflicts (above), not duplicates.
- **Knowledge-base tier** (`kb_tier=` in notes, from file B's `source` column; config `kb_tiers`):
  `tier1_official` = source is `https://agripunjab.punjab.gov.pk/urdu/` (the site's home page only —
  the exact page is not given); `safety` = "safe refusal" (moved to `safety_refusals.jsonl`);
  `tier2_needs_expert` = everything else: general plant pathology / agronomy, CIMMYT, news articles,
  `existing_dataset` (urdu_agriculture_Question_Answer.xlsx), and all file-A rows (no source column).
  Even tier 1 is unverified until matched against the page (`KB_CANDIDATES.md`).
- **Fields**: `crop=wheat`; `lang` = the column's language; `topic` from the file's `category` via
  the config map (categories with no crop topic, e.g. `yield_loss`, use a keyword guess);
  `season`: rabi→rabi, all_seasons→any, kharif→any; `source_type=llm`; `expert_verified=false`;
  `kb_chunk_ids=[]`; `annotator=import:wheat_qa_v1`.

## `notes` tokens (`; `-separated)

| Token | Meaning |
|---|---|
| `group=B:<group_id>` / `group=A:<id>` | Split unit. The 3 language records of a row share it; variants share it with their v1 row. |
| `src=<file>/<sheet>/row<n>/id<id>` | Source cell row (Excel row number). |
| `category=<value>` | The file's own category. |
| `kb_tier=<tier>`, `kb_source=<value>` | Knowledge-base tier and the file's `source` value (see Rules). |
| `synthetic`, `provenance unverified; file label: <value>` | On every record; the file's `source_type` is kept here, not trusted. |
| `pesticide_check` | An answer (any language) names a pesticide — needs expert check. 66 v1 records. |
| `code_mixed` | ≥ 10 % of the words are in the other script. 19 v1 records. |
| `season_suspect` | File said `kharif` (wheat is rabi); season set to `any`. 0 v1 records (all were template fill-ins). |
| `topic_from_keywords` | The category has no crop topic; topic is the keyword guess. |
| `topic_keyword_disagrees=<topic>` | The category's topic was kept, but keywords suggest another. |
| `case_template` | Question came from a `(Case N)` template. |
| `augmentation_only`, `variant_type=<value>` | Variants file only. |

## Using it later

- Split by `group`, never by record, so translations and variants of one Q&A stay together.
- Variants may only join **train**, and only for groups whose v1 records are in train.
- `zarai.data.split_qa` puts only `farmer`/`expert` records in test, so v1 (all `llm`) can never fill
  a test split — by design.
