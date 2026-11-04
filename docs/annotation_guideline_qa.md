# Q&A annotation guideline (template)

## Purpose
Build expert-checked question and answer pairs for training and evaluating ZarAI. Every record follows `src/zarai/data/schemas/qa_record.schema.json`.

## Fields

| Field | How to fill it |
|---|---|
| `id` | `qa_0001` and upward, never reused |
| `lang` | `ur` (Urdu script), `ur-Latn` (Roman Urdu), `en` (English) |
| `crop` | wheat, potato, tomato, sugarcane, maize, cotton, rice, general |
| `topic` | See the topic list |
| `season` | rabi, kharif, any |
| `question` | The farmer's words, spelling errors included |
| `answer` | Supported only by the listed chunks |
| `kb_chunk_ids` | Chunks that support the answer; empty for off-topic questions |
| `source_type` | farmer, expert, llm. Only farmer and expert can enter the test set |
| `annotator` | Your ID |
| `expert_verified`, `verifier` | Set after expert review |
| `notes` | Anything unusual |

## Topic list
`sowing_varieties`, `fertiliser`, `irrigation`, `pest`, `disease`, `weed`, `harvest_storage`, `off_topic_unsafe`.

## Examples per language
Replace these placeholders with expert-approved examples.

- **Urdu:** سوال: [QUESTION] / جواب: [ANSWER]
- **Roman Urdu:** Sawal: [QUESTION] / Jawab: [ANSWER]
- **English:** Question: [QUESTION] / Answer: [ANSWER]

## Edge cases
- Mixed language: use the language of most of the words.
- Roman Urdu spelling varies; keep it as written.
- More than one crop or topic: split into separate records.
- Answer not in the KB: flag it for RA-3 rather than writing from memory.

## Unsafe questions
Label these `off_topic_unsafe`: requests for doses above label limits, banned chemicals, or use of pesticides on people or animals. The answer should refuse and point to the extension officer.

Every dose or pesticide name in an answer must appear in a listed chunk.

## Double annotation
A subset is annotated twice, then scored with `python -m zarai.data.agreement`. Agreement target: [KAPPA].
