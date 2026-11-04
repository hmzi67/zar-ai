"""Validate JSONL/CSV files against the Q&A or image metadata schema and print an error report."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from zarai.common.io import read_csv, read_jsonl

SCHEMA_DIR = Path(__file__).parent / "schemas"
ID_FIELD = {"qa_record": "id", "image_meta": "image_id"}
BOOLEANS = {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}


def load_schema(name: str) -> dict:
    return json.loads((SCHEMA_DIR / f"{name}.schema.json").read_text(encoding="utf-8"))


def coerce_csv_row(row: dict, schema: dict) -> dict:
    """CSV cells are all strings: convert them to the types the schema expects."""
    out = {}
    for key, value in row.items():
        spec = schema["properties"].get(key, {})
        if spec.get("type") == "array":
            out[key] = [v.strip() for v in (value or "").split(";") if v.strip()]
        elif not value:
            continue  # blank cell = missing, so `required` reports it
        elif spec.get("type") == "boolean":
            out[key] = BOOLEANS.get(value.strip().lower(), value)  # bad value fails validation
        else:
            out[key] = value
    return out


def validate_records(records: list[dict], schema_name: str) -> list[str]:
    """Return one error message per problem; an empty list means the records are valid."""
    validator = Draft202012Validator(load_schema(schema_name), format_checker=FormatChecker())
    errors = []
    for i, record in enumerate(records, start=1):
        for error in validator.iter_errors(record):
            field = ".".join(str(p) for p in error.absolute_path) or "<record>"
            errors.append(f"record {i}: {field}: {error.message}")
    counts = Counter(r.get(ID_FIELD[schema_name]) for r in records)
    errors += [f"duplicate id: {k}" for k, n in counts.items() if k is not None and n > 1]
    return errors


def validate_file(path: str | Path, schema_name: str) -> list[str]:
    path = Path(path)
    if path.suffix == ".csv":
        schema = load_schema(schema_name)
        records = [coerce_csv_row(row, schema) for row in read_csv(path)]
    else:
        records = read_jsonl(path)
    return validate_records(records, schema_name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", required=True, choices=sorted(ID_FIELD))
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    failed = False
    for path in args.files:
        errors = validate_file(path, args.schema)
        print(f"{path}: {'OK' if not errors else f'{len(errors)} error(s)'}")
        for error in errors:
            print(f"  {error}")
        failed = failed or bool(errors)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
