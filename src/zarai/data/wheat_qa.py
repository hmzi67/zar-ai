"""Build wheat Q&A v1 from the two synthetic Excel files in data/qa/.

Each Excel row holds one Q&A in Urdu, Roman Urdu and English. It becomes three schema records
(`ur`, `ur-Latn`, `en`) that share `group=` in `notes`, so a later split can keep them together.
Answers are only Unicode-normalised: their wording is never changed.

Steps: normalise text, redact phone/CNIC/e-mail, strip "(Case N)" template suffixes, quarantine
unsafe answers, conflicting-answer groups and generic answers given to questions of another topic
for expert review, keep one original per file-B group
(the other rows of the group become train-only variants), drop duplicates and near-duplicates
(file B wins over file A), map category→topic and season, flag pesticides and code-mixed text.
"""

import argparse
import csv
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf
from openpyxl import load_workbook
from rapidfuzz import fuzz

from zarai.common.io import write_json, write_jsonl
from zarai.common.logging import get_logger
from zarai.data.validate import validate_records
from zarai.nlp.normalise import match_key, normalise_text

LANG_SUFFIX = {"ur": "urdu", "ur-Latn": "roman_urdu", "en": "english"}
TEXT_COLUMNS = [f"{kind}_{s}" for kind in ("question", "answer") for s in LANG_SUFFIX.values()]
COLUMN_LANG = {
    f"{kind}_{s}": lang for lang, s in LANG_SUFFIX.items() for kind in ("question", "answer")
}
LOG_COLUMNS = [
    "action",
    "reason",
    "file",
    "sheet",
    "excel_row",
    "source_id",
    "group",
    "detail",
    "kept",
]
REVIEW_COLUMNS = [
    "reason", "detail", "conflict_key", "file", "sheet", "excel_row", "source_id", "group",
    "variant_type", "variants_excluded", "source_rows", "kb_tier", "source", "category", *TEXT_COLUMNS, "expert_decision", "expert_notes",
]  # fmt: skip

# "(Case 31)" or "(کیس 31)" at the end of a question, before any final ? ؟ or full stop.
CASE_SUFFIX = re.compile(r"\s*\((?:Case|case|کیس)\s*[\d۰-۹]+\)(?=\s*[?؟.۔]?\s*$)")
ARABIC = re.compile("[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
URDU_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
PII = {
    "phone": re.compile(r"(?:\+92|0092|\b0)\s?3\d{2}[\s-]?\d{7}\b"),
    "cnic": re.compile(r"\b\d{5}-\d{7}-\d\b"),
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
}

# Active ingredients, a few Pakistani brand names and their Urdu spellings. Sulphur is left out
# because the files only mention it as a nutrient.
PESTICIDES = """tebuconazole propiconazole carboxin carbendazim thiophanate mancozeb azoxystrobin
difenoconazole triadimefon triadimenol thiram imidacloprid chlorpyrifos cyhalothrin cypermethrin
deltamethrin bifenthrin emamectin fipronil dimethoate acetamiprid thiamethoxam malathion profenofos
glyphosate isoproturon metribuzin pendimethalin clodinafop fenoxaprop mesosulfuron iodosulfuron
sulfosulfuron bromoxynil carfentrazone pinoxaden tribenuron metsulfuron phosphide phosphine
hexaconazole trifloxystrobin flutriafol cyproconazole epoxiconazole prothioconazole metalaxyl
tricyclazole trichoderma folicur nativo topsin vitavax confidor
ٹیبوکونازول پروپیکونازول کاربوکسن کاربینڈازم مینکوزیب امیڈاکلوپرڈ کلورپائریفاس گلائفوسیٹ فاسفائیڈ
ٹریکوڈرما تھائیرم""".split() + ["methyl bromide", "copper oxychloride", "2,4-d", "mcpa"]
PESTICIDE_RE = re.compile(
    "|".join(rf"(?<![^\W\d_]){re.escape(p)}(?![^\W\d_])" for p in PESTICIDES), re.IGNORECASE
)

# Keyword topic guess over question (x2) and answer (x1); ties go to the earlier topic.
TOPIC_KEYWORDS = {
    "pest": r"\bpests?\b|insect|aphid|termite|army ?worm|borer|\bmites?\b|rodent|\brats?\b|caterpillar|larva|keer[ae]|sundi|\btela\b|deemak|کیڑ|سنڈی|تیلا|دیمک|چوہ",
    "disease": r"disease|\brust\b|mildew|blight|\bbunt\b|smut|septoria|tan spot|leaf spot|fung|\bblast\b|virus|\brot\b|pathogen|bimari|bemari|\bzang\b|kangiyari|بیماری|زنگ|کانگیاری|پھپھوند|فنگ|جھلساؤ",
    "weed": r"\bweeds?\b|herbicide|weedicide|dumbi|jangli ja[ie]|phalaris|wild oat|جڑی بوٹی|دمبی|جنگلی جئی|بوٹی مار",
    "fertiliser": r"fertili[sz]|\burea\b|\bdap\b|nitrogen|phosph|potash|potassium|\bzinc\b|boron|sulphate|sulfate|\bnpk\b|manure|compost|nutrient|deficien|soil test|gypsum|khaa?d\b|کھاد|یوریا|نائٹروجن|فاسفورس|پوٹاش|زنک|گوبر",
    "irrigation": r"irrigat|\bwater|drought|waterlog|sprinkler|\bdrip\b|paa?ni|senchai|aabpashi|پانی|آبپاشی|سنچائی|آبیاری|خشک سالی",
    "harvest_storage": r"harvest|thresh|storage|\bstore|godown|warehouse|combine|post.harvest|grain moisture|katai|kataai|zakhee?ra|gahai|کٹائی|ذخیر|گہائی|گودام",
    "sowing_varieties": r"\bsow|variet|seed rate|planting|land prep|seedbed|germinat|certified seed|\bdrill|bawai|bowai|aqsaa?m|بوائی|کاشت|اقسام|قسم|شرح بیج",
}
TOPIC_RE = {topic: re.compile(pattern, re.IGNORECASE) for topic, pattern in TOPIC_KEYWORDS.items()}
# An answer given to questions of at least this many crop topics is a generic template answer.
GENERIC_ANSWER_MIN_TOPICS = 2

log = logging.getLogger(__name__)


@dataclass
class Result:
    v1: list[dict]
    variants: list[dict]
    safety: list[dict]
    review: list[dict]
    log: list[dict]
    summary: dict


# --------------------------------------------------------------------------- small helpers
def read_workbook(path: str | Path, file_key: str) -> list[dict]:
    """All sheets of one workbook as dicts of strings, with file, sheet and Excel row number."""
    workbook = load_workbook(path, read_only=True, data_only=True)
    rows = []
    for sheet in workbook.worksheets:
        values = sheet.iter_rows(values_only=True)
        header = [str(h).strip() if h is not None else "" for h in next(values)]
        for excel_row, cells in enumerate(values, start=2):
            row = {h: "" if v is None else str(v) for h, v in zip(header, cells, strict=False) if h}
            rows.append({"file": file_key, "sheet": sheet.title, "excel_row": excel_row, **row})
    workbook.close()
    return rows


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Replace phone numbers, CNICs and e-mails with [PHONE]/[CNIC]/[EMAIL]. Urdu digits count."""
    found = []
    for kind, pattern in PII.items():
        # Urdu digits map one-to-one onto ASCII, so match positions in the copy fit the original.
        spans = [m.span() for m in pattern.finditer(text.translate(URDU_DIGITS))]
        for start, end in reversed(spans):
            text = text[:start] + f"[{kind.upper()}]" + text[end:]
        found += [kind] * len(spans)
    return text, found


def is_code_mixed(text: str, lang: str) -> bool:
    """True if ≥ 10 % of the words are in the other script (Latin in Urdu, Arabic in Roman/English)."""
    words = re.findall(r"\w+", text)
    arabic = sum(bool(ARABIC.search(w)) for w in words)
    latin = sum(bool(re.search(r"[A-Za-z]", w)) for w in words)
    foreign = latin if lang == "ur" else arabic
    return arabic + latin > 0 and foreign / (arabic + latin) >= 0.10


def keyword_topic(question: str, answer: str) -> str:
    score = {t: 2 * len(r.findall(question)) + len(r.findall(answer)) for t, r in TOPIC_RE.items()}
    best = max(score.values())
    return next(t for t in TOPIC_RE if score[t] == best) if best else "other"


def row_text(row: dict, kind: str) -> str:
    return " ".join(row[f"{kind}_{s}"] for s in LANG_SUFFIX.values())


def compile_rules(rules) -> list[dict]:
    """Config quarantine rules → dicts with compiled `question` / `answer` patterns."""
    compiled = []
    for rule in rules:
        if not (rule.get("question") or rule.get("answer")):
            raise ValueError(f"quarantine rule {rule.name!r} needs a question or answer pattern")
        compiled.append({
            "name": rule.name, "reason": rule.reason, "detail": rule.detail,
            **{k: re.compile(rule[k], re.IGNORECASE) for k in ("question", "answer") if rule.get(k)},
        })  # fmt: skip
    return compiled


def rule_matches(rule: dict, row: dict) -> bool:
    """True if every pattern of the rule matches at least one language cell of its kind."""
    return all(
        any(rule[kind].search(row[f"{kind}_{s}"]) for s in LANG_SUFFIX.values())
        for kind in ("question", "answer")
        if kind in rule
    )


def kb_tier(row: dict, cfg) -> tuple[str, str]:
    """(tier, source) from file B's `source` column; file A has no source and needs an expert."""
    source = row.get("source", "") if row["file"] == "B" else ""
    for rule in cfg.kb_tiers:
        if source and rule.match.lower() in source.lower():
            return rule.tier, source
    return cfg.kb_tier_default, source or "none (file A has no source column)"


# --------------------------------------------------------------------------- pipeline
def process(rows: list[dict], cfg) -> Result:
    """Rows from read_workbook (file "A" and "B") → v1 records, variants, review rows, log, summary."""
    actions: list[dict] = []

    def note(action, row, reason, detail="", kept=""):
        actions.append({
            "action": action, "reason": reason, "file": row["file"], "sheet": row["sheet"],
            "excel_row": row["excel_row"], "source_id": row.get("id", ""), "group": row["group"],
            "detail": detail, "kept": kept,
        })  # fmt: skip

    counts = Counter()

    # 1. Normalise, redact, strip "(Case N)"; drop rows without any question or answer.
    usable = []
    for row in rows:
        row["group"] = (
            f"B:{row['group_id']}" if row.get("group_id") else f"{row['file']}:{row.get('id')}"
        )
        row["flags"] = []
        for col in TEXT_COLUMNS:
            raw = row.get(col, "")
            text = normalise_text(raw, COLUMN_LANG[col])
            if text != raw:
                counts["cells_normalised"] += 1
                note("normalise", row, "unicode/space/punctuation", f"{col}: {raw!r} -> {text!r}")
            text, pii = redact_pii(text)
            if pii:
                counts["pii_redacted"] += len(pii)
                row["flags"].append("pii_redacted")
                note("redact", row, "pii", f"{col}: {', '.join(pii)}")
            if col.startswith("question") and CASE_SUFFIX.search(text):
                text = CASE_SUFFIX.sub("", text)
                if "case_template" not in row["flags"]:
                    row["flags"].append("case_template")
                    counts["case_suffix_rows"] += 1
            row[col] = text
        if not any(row[c] for c in TEXT_COLUMNS if c.startswith("question")) or not any(
            row[c] for c in TEXT_COLUMNS if c.startswith("answer")
        ):
            counts["empty_rows"] += 1
            note("drop", row, "empty_row")
            continue
        row["qk"], row["ak"] = match_key(row["question_english"]), match_key(row["answer_english"])
        usable.append(row)
    if counts["case_suffix_rows"]:
        log.info("stripped (Case N) from %d rows", counts["case_suffix_rows"])

    by_group = defaultdict(list)
    for row in usable:
        by_group[row["group"]].append(row)
    is_base = [r for r in usable if r["file"] == "A" or r.get("variant_type") == "original"]

    def question_topic(row) -> str:
        """The row's topic from its category, or the question keywords if the category has none."""
        category = row.get("category", "")
        if category not in cfg.category_topic:
            raise ValueError(
                f"category {category!r} is not in category_topic; add it to the config"
            )
        return cfg.category_topic[category] or keyword_topic(row_text(row, "question"), "")

    # 2. Quarantine. First the config rules in order (unsafe, scientifically wrong, template
    # fill-ins, named varieties); the first rule that matches a row names the reason.
    quarantined: dict[str, tuple[str, str, str]] = {}  # group -> (reason, detail, conflict key)
    rules = compile_rules(cfg.quarantine_rules)
    for row in usable:
        if row["group"] in quarantined:
            continue
        for rule in rules:
            if rule_matches(rule, row):
                quarantined[row["group"]] = (rule["reason"], rule["detail"], rule["name"])
                break

    # Then questions that carry more than one answer. Answers of one question that are
    # near-identical (ratio ≥ answer_paraphrase_threshold) are one answer in two wordings: they
    # get a shared key so the duplicate step keeps only one of them.
    answers_by_q = defaultdict(list)
    for row in is_base:
        if row["group"] not in quarantined and row["ak"] not in answers_by_q[row["qk"]]:
            answers_by_q[row["qk"]].append(row["ak"])
    canonical = {}
    for qk, answers in answers_by_q.items():
        clusters: list[str] = []
        for ak in answers:
            same = next(
                (c for c in clusters if fuzz.ratio(c, ak) >= cfg.answer_paraphrase_threshold), None
            )
            canonical[qk, ak] = same or ak
            if same is None:
                clusters.append(ak)
        answers_by_q[qk] = clusters
    for row in usable:
        if (row["qk"], row["ak"]) in canonical and canonical[row["qk"], row["ak"]] != row["ak"]:
            note(
                "merge",
                row,
                "answer_paraphrase",
                f"answer treated as same as: {canonical[row['qk'], row['ak']]}",
            )
            row["ak"] = canonical[row["qk"], row["ak"]]
            row["paraphrase"] = True
    conflict_keys = {q for q, answers in answers_by_q.items() if len(answers) > 1}
    for row in is_base:
        if row["qk"] in conflict_keys and row["group"] not in quarantined:
            n = len(answers_by_q[row["qk"]])
            quarantined[row["group"]] = ("conflicting_answers", f"{n} different answers", row["qk"])

    # Question/answer mismatch: a generic answer that the files give to questions of several
    # topics, used here for a question whose topic it never mentions ("best practices for weed
    # management" → NPK advice). Keyword absence alone is not enough: specific answers often
    # skip the topic word ("how to recognise rust?" → "yellow stripes on the leaf").
    topics_by_answer = defaultdict(set)
    for row in is_base:
        topics_by_answer[row["ak"]].add(question_topic(row))
    for row in is_base:
        topic = question_topic(row)
        answer_topics = sorted(topics_by_answer[row["ak"]] - {"other", "off_topic_unsafe"})
        if (
            row["group"] not in quarantined
            and len(answer_topics) >= GENERIC_ANSWER_MIN_TOPICS
            and not (topic in TOPIC_RE and TOPIC_RE[topic].search(row_text(row, "answer")))
        ):
            detail = (
                f"question topic {topic}; answer is generic (given to {', '.join(answer_topics)} "
                f"questions) and never mentions {topic}"
            )
            quarantined[row["group"]] = ("question_answer_mismatch", detail, row["qk"])

    # Conflict and unsafe rows are listed one by one; mismatch rows once per distinct question
    # (the "(Case N)" copies of one template question would otherwise fill the sheet).
    review, mismatch_entries = [], {}
    for group, (reason, detail, key) in quarantined.items():
        members = by_group[group]
        excluded = sum(r.get("variant_type") not in (None, "", "original") for r in members)
        for row in members:
            note("quarantine", row, reason, detail)
            if row["file"] == "A" or row.get("variant_type") == "original":
                if reason == "question_answer_mismatch" and key in mismatch_entries:
                    mismatch_entries[key]["source_rows"] += 1
                    mismatch_entries[key]["variants_excluded"] += excluded
                    continue
                review.append({
                    "reason": reason, "detail": detail, "conflict_key": key,
                    **{c: row.get(c, "") for c in ("file", "sheet", "excel_row", "group", "variant_type", "category", *TEXT_COLUMNS)},
                    "source_id": row.get("id", ""), "variants_excluded": excluded, "source_rows": 1,
                    "kb_tier": kb_tier(row, cfg)[0], "source": kb_tier(row, cfg)[1],
                    "expert_decision": "", "expert_notes": "",
                })  # fmt: skip
                if reason == "question_answer_mismatch":
                    mismatch_entries[key] = review[-1]
    review.sort(key=lambda r: (r["reason"], r["conflict_key"], r["file"], int(r["excel_row"])))
    counts["quarantined_rows"] = sum(len(by_group[g]) for g in quarantined)

    # 3. One original per file-B group goes to v1; the rest of the group become variants.
    def id_order(row):
        return int(row["id"]) if str(row.get("id", "")).isdigit() else 0

    chosen, variants_rows, b_anchors = [], [], defaultdict(list)
    for group, members in by_group.items():
        if not group.startswith("B:") or group in quarantined:
            continue
        originals = sorted(
            (r for r in members if r.get("variant_type") == "original"), key=id_order
        )
        first = originals[0] if originals else min(members, key=id_order)
        chosen.append(first)
        for row in originals:
            b_anchors[row["ak"]].append(row)
        for row in sorted(members, key=id_order):
            if row is not first:
                variants_rows.append(row)
                if row.get("variant_type") == "original":
                    note("to_variants", row, "extra_original_paraphrase", kept=first.get("id", ""))
    chosen.sort(key=id_order)

    # 4. Duplicates: exact (question, answer) or same answer with question ratio ≥ threshold.
    threshold = cfg.near_duplicate_threshold
    kept, seen = [], defaultdict(list)

    def match(row, candidates):
        best = None
        for other in candidates:
            score = 100 if other["qk"] == row["qk"] else fuzz.ratio(other["qk"], row["qk"])
            if score >= threshold and (best is None or score > best[0]):
                best = (score, other)
        return best

    for row in chosen + [r for r in usable if r["file"] == "A" and r["group"] not in quarantined]:
        hit = None
        if row["file"] == "A":
            hit = match(row, b_anchors[row["ak"]])
            where = "B"
        if hit is None:
            hit = match(row, seen[row["ak"]])
            where = row["file"]
        if hit:
            score, other = hit
            if score == 100:
                if row.get("paraphrase") or other.get("paraphrase"):
                    reason = "answer_paraphrase"
                elif "case_template" in row["flags"]:
                    reason = "case_template_collapsed"
                else:
                    reason = "exact_duplicate"
            else:
                reason = "near_duplicate"
            reason += f"_of_{'B' if where == 'B' or other['file'] == 'B' else 'A'}"
            counts[f"drop_{reason}"] += 1
            note("drop", row, reason, f"ratio={score}", f"{other['file']}:{other.get('id')}")
            continue
        seen[row["ak"]].append(row)
        kept.append(row)

    # 5. Rows → records.
    def annotate(row) -> tuple[str, str, list[str]]:
        flags = list(row["flags"])
        guess = keyword_topic(row_text(row, "question"), row_text(row, "answer"))
        topic = cfg.category_topic[row.get("category", "")]  # validated by question_topic
        if topic is None:
            topic = guess
            flags.append("topic_from_keywords")
        elif guess not in ("other", topic) and topic != "off_topic_unsafe":
            flags.append(f"topic_keyword_disagrees={guess}")
        if row.get("season") not in cfg.season:
            raise ValueError(f"season {row.get('season')!r} is not in the season map")
        season = cfg.season[row["season"]]
        if row["season"] == "kharif":
            flags.append("season_suspect")
        if PESTICIDE_RE.search(row_text(row, "answer")):
            flags.append("pesticide_check")
        return topic, season, flags

    def to_records(row, extra_flags=()):
        topic, season, flags = annotate(row)
        tier, source = kb_tier(row, cfg)
        for lang, suffix in LANG_SUFFIX.items():
            question, answer = row[f"question_{suffix}"], row[f"answer_{suffix}"]
            if not question or not answer:
                note("drop", row, "empty_cell", lang)
                continue
            mixed = is_code_mixed(question, lang) or is_code_mixed(answer, lang)
            notes = [
                f"group={row['group']}",
                f"src={row['file']}/{row['sheet']}/row{row['excel_row']}/id{row.get('id', '')}",
                f"category={row.get('category', '')}",
                f"kb_tier={tier}", f"kb_source={source}",
                *flags, *(["code_mixed"] if mixed else []), *extra_flags,
                "synthetic", f"provenance unverified; file label: {row.get('source_type', '')}",
            ]  # fmt: skip
            yield row, {
                "id": "", "lang": lang, "crop": "wheat", "topic": topic, "season": season,
                "question": question, "answer": answer, "kb_chunk_ids": [], "source_type": "llm",
                "annotator": cfg.annotator, "expert_verified": False, "verifier": None,
                "notes": "; ".join(notes),
            }  # fmt: skip

    record_keys = set()

    def unique(pairs):
        for row, record in pairs:
            key = (record["lang"], match_key(record["question"]), match_key(record["answer"]))
            if key in record_keys:
                counts["drop_record_duplicate"] += 1
                note("drop", row, "record_duplicate", record["lang"])
                continue
            record_keys.add(key)
            yield record

    v1 = list(unique(p for row in kept for p in to_records(row)))
    variants = list(
        unique(
            p
            for row in variants_rows
            for p in to_records(
                row, ("augmentation_only", f"variant_type={row.get('variant_type')}")
            )
        )
    )

    # Safe-refusal rows are not crop knowledge: they go to their own file, variants included.
    def is_safety(record):
        return "kb_tier=safety" in record["notes"].split("; ")

    safety = [r for r in v1 + variants if is_safety(r)]
    v1 = [r for r in v1 if not is_safety(r)]
    variants = [r for r in variants if not is_safety(r)]
    for i, record in enumerate(v1, start=1):
        record["id"] = f"qa_{i:05d}"
    for i, record in enumerate(variants, start=cfg.variant_id_start):
        record["id"] = f"qa_{i}"
    for i, record in enumerate(safety, start=cfg.safety_id_start):
        record["id"] = f"qa_{i}"

    summary = summarise(
        rows,
        quarantined,
        by_group,
        variants_rows,
        kept,
        v1,
        variants,
        safety,
        review,
        actions,
        counts,
    )
    return Result(v1, variants, safety, review, actions, summary)


def summarise(
    rows, quarantined, by_group, variants_rows, kept, v1, variants, safety, review, actions, counts
):
    def flag_count(records, flag):
        return sum(flag in r["notes"].split("; ") for r in records)

    def tier_of(record):
        return next(t[8:] for t in record["notes"].split("; ") if t.startswith("kb_tier="))

    safety_groups = {r["notes"].split("; ")[0][6:] for r in safety}
    quarantine = {}
    for reason in sorted({q[0] for q in quarantined.values()}):
        groups = [g for g, q in quarantined.items() if q[0] == reason]
        quarantine[reason] = {
            "rules_or_questions": len({quarantined[g][2] for g in groups}),
            "groups": len(groups),
            "rows_excluded": dict(Counter(r["file"] for g in groups for r in by_group[g])),
            "review_csv_rows": sum(r["reason"] == reason for r in review),
        }
    # Where every input row ended up (row-level actions only; record-level drops are separate).
    accounting = {}
    for file in sorted({r["file"] for r in rows}):
        dropped = Counter(
            a["reason"] for a in actions
            if a["file"] == file and a["action"] == "drop" and a["reason"] not in ("record_duplicate", "empty_cell")
        )  # fmt: skip
        mine = [r for r in kept if r["file"] == file]
        moved = [r for r in variants_rows if r["file"] == file]
        accounting[file] = {
            "input": sum(r["file"] == file for r in rows),
            "quarantined": sum(r["file"] == file for g in quarantined for r in by_group[g]),
            "dropped": dict(dropped),
            "to_variants": sum(r["group"] not in safety_groups for r in moved),
            "to_safety": sum(r["group"] in safety_groups for r in mine + moved),
            "kept_in_v1": sum(r["group"] not in safety_groups for r in mine),
        }

    def by_tier(records):
        return dict(sorted(Counter(tier_of(r) for r in records).items()))

    return {
        "warning": "Synthetic, template-generated data with unverified provenance. Not a test set "
        "and not knowledge-base ground truth.",
        "input_rows": dict(Counter(r["file"] for r in rows)),
        "empty_rows_dropped": counts["empty_rows"],
        "cells_normalised": counts["cells_normalised"],
        "pii_redactions": counts["pii_redacted"],
        "case_template_rows": counts["case_suffix_rows"],
        "quarantine": quarantine | {"review_csv_rows": len(review)},
        "row_accounting": accounting,
        "record_duplicates_dropped": counts["drop_record_duplicate"],
        "kb_tier_records": {
            "v1": by_tier(v1), "variants": by_tier(variants), "safety_refusals": by_tier(safety),
            "review_csv_rows": dict(sorted(Counter(r["kb_tier"] for r in review).items())),
        },  # fmt: skip
        "v1": {
            "rows": len(v1) // len(LANG_SUFFIX),
            "rows_by_file": dict(
                Counter(r["file"] for r in kept if r["group"] not in safety_groups)
            ),
            "records": len(v1),
            "by_lang": dict(Counter(r["lang"] for r in v1)),
            "by_topic": dict(Counter(r["topic"] for r in v1).most_common()),
            "by_season": dict(Counter(r["season"] for r in v1)),
            "flags": {
                f: flag_count(v1, f)
                for f in ("pesticide_check", "code_mixed", "season_suspect", "topic_from_keywords", "case_template", "pii_redacted")
            }  # fmt: skip
            | {
                "topic_keyword_disagrees": sum("topic_keyword_disagrees=" in r["notes"] for r in v1)
            },
        },
        "variants": {
            "records": len(variants),
            "groups": len({r["notes"].split("; ")[0] for r in variants}),
            "by_lang": dict(Counter(r["lang"] for r in variants)),
        },
        "safety_refusals": {
            "records": len(safety),
            "groups": len(safety_groups),
            "base_records": sum("augmentation_only" not in r["notes"].split("; ") for r in safety),
        },
    }


# --------------------------------------------------------------------------- I/O
def write_csv(path: str | Path, rows: list[dict], columns: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:  # -sig so Excel shows Urdu
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def kb_candidates_markdown(v1: list[dict], review: list[dict]) -> str:
    """Checklist of tier-1 (agripunjab) v1 rows, to match one by one against the source site."""

    def token(record, key):
        return next(
            (t.split("=", 1)[1] for t in record["notes"].split("; ") if t.startswith(key + "=")), ""
        )

    groups: dict[str, dict] = defaultdict(dict)
    for record in v1:
        if token(record, "kb_tier") == "tier1_official":
            groups[token(record, "group")][record["lang"]] = record
    sources = sorted({token(r, "kb_source") for g in groups.values() for r in g.values()})
    lines = [
        "# KB candidates — tier 1 (agripunjab)",
        "",
        "> Generated by `zarai.data.wheat_qa`; do not edit by hand. **Synthetic data, unverified.**",
        "> Each row below is a file-B group whose `source` column cites agripunjab. Nothing here is",
        "> knowledge-base ground truth until it has been matched against the original page.",
        "",
        f"- Rows: **{len(groups)}** (each is 3 records in `wheat_qa_v1.jsonl`: ur, ur-Latn, en).",
        f"- Source URLs cited: {', '.join(f'`{s}`' for s in sources)}.",
        "  The file gives only the site's Urdu home page, not the page each answer came from, so the",
        "  page has to be found by searching the site for the Urdu answer.",
        "",
    ]
    for i, (group, by_lang) in enumerate(sorted(groups.items()), start=1):
        en, ur = by_lang.get("en", {}), by_lang.get("ur", {})
        any_record = en or ur
        flags = [
            t for t in any_record["notes"].split("; ")
            if t in ("pesticide_check", "code_mixed", "season_suspect") or t.startswith("topic_")
        ]  # fmt: skip
        lines += [
            f"## {i}. `{group}` — {any_record['topic']}",
            "",
            f"- Records: {', '.join(by_lang[lang]['id'] for lang in ('ur', 'ur-Latn', 'en') if lang in by_lang)}",
            f"- Source: {token(any_record, 'kb_source')} (cell `{token(any_record, 'src')}`)",
            *([f"- Flags: {', '.join(flags)}"] if flags else []),
            f"- **Q (ur):** {ur.get('question', '')}",
            f"- **A (ur):** {ur.get('answer', '')}",
            f"- **Q (en):** {en.get('question', '')}",
            f"- **A (en):** {en.get('answer', '')}",
            "- [ ] Found on agripunjab — page URL: ",
            "- [ ] Answer matches the page",
            "",
        ]
    held = [r for r in review if r["kb_tier"] == "tier1_official"]
    if held:
        lines += [
            "## Tier-1 rows held back in `data/interim/qa_review.csv`",
            "",
            "These cite agripunjab too but were quarantined, so they are not in v1.",
            "",
            "| Group | Reason | Question (en) |",
            "|---|---|---|",
            *(f"| `{r['group']}` | {r['reason']} | {r['question_english']} |" for r in held),
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data/wheat_qa_v1.yaml"))
    args = parser.parse_args()
    get_logger(__name__)
    cfg = OmegaConf.load(args.config)

    rows = [row for key, path in cfg.inputs.items() for row in read_workbook(path, key)]
    result = process(rows, cfg)
    for name, records in (
        ("v1", result.v1),
        ("variants", result.variants),
        ("safety", result.safety),
    ):
        errors = validate_records(records, "qa_record")
        if errors:
            raise SystemExit(f"{name}: {len(errors)} schema errors, first: {errors[:5]}")

    out = cfg.outputs
    write_jsonl(out.v1, result.v1)
    write_jsonl(out.variants, result.variants)
    write_jsonl(out.safety, result.safety)
    write_json(out.summary, result.summary)
    write_csv(out.review, result.review, REVIEW_COLUMNS)
    write_csv(out.log, result.log, LOG_COLUMNS)
    Path(out.kb_candidates).write_text(
        kb_candidates_markdown(result.v1, result.review), encoding="utf-8"
    )
    log.info("v1: %d records; variants: %d; safety: %d; review: %d rows",
             len(result.v1), len(result.variants), len(result.safety), len(result.review))  # fmt: skip


if __name__ == "__main__":
    main()
