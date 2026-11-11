"""Urdu OCR for PDFs whose text layer is missing or unusable.

Many Urdu government PDFs embed fonts without a Unicode map, so pypdf returns only
digits and Latin words. This module renders each page and runs Tesseract on it.

System requirements (not pip):
    apt-get install -y tesseract-ocr tesseract-ocr-urd poppler-utils

Python: pytesseract, pdf2image (see requirements/nlp.txt).

    python -m zarai.kb.ocr --input data/raw/kb_sources/src_0001.pdf \
        --out data/interim/kb_text/src_0001.txt --report data/interim/kb_text/src_0001.ocr.csv
"""

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

from zarai.common.logging import get_logger

URDU_LETTER = re.compile(r"[؀-ۿ]")
LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
PAGE_SEPARATOR = "\n\n"


@dataclass
class PageResult:
    page: int
    text: str
    mean_conf: float  # Tesseract word confidence, 0-100; -1 if no words were found
    n_words: int


def urdu_share(text: str) -> float:
    """Share of letters that are Arabic-script. Digits and punctuation are ignored."""
    letters = LETTER.findall(text)
    if not letters:
        return 0.0
    return sum(1 for ch in letters if URDU_LETTER.match(ch)) / len(letters)


def needs_ocr(text: str, language: str, min_urdu_share: float = 0.3, min_chars: int = 200) -> bool:
    """True when an extracted text layer is too thin or, for Urdu sources, has almost no Urdu."""
    if len(text.strip()) < min_chars:
        return True
    return language.startswith("ur") and urdu_share(text) < min_urdu_share


def _ocr_image(image, lang: str, psm: int) -> PageResult:
    import pytesseract

    config = f"--psm {psm}"
    text = pytesseract.image_to_string(image, lang=lang, config=config)
    data = pytesseract.image_to_data(
        image, lang=lang, config=config, output_type=pytesseract.Output.DICT
    )
    confs = [
        float(c)
        for c, w in zip(data["conf"], data["text"], strict=True)
        if w.strip() and float(c) >= 0
    ]
    mean_conf = sum(confs) / len(confs) if confs else -1.0
    return PageResult(page=0, text=text.strip(), mean_conf=mean_conf, n_words=len(confs))


def ocr_pdf_pages(
    path: str | Path, lang: str = "urd+eng", dpi: int = 300, psm: int = 4
) -> list[PageResult]:
    """OCR every page. psm 4 (single column of variable sizes) suits the tables in AARI plans."""
    from pdf2image import convert_from_path

    results = []
    for i, image in enumerate(convert_from_path(str(path), dpi=dpi), start=1):
        result = _ocr_image(image, lang, psm)
        result.page = i
        results.append(result)
    return results


def ocr_pdf(path: str | Path, lang: str = "urd+eng", dpi: int = 300) -> str:
    """Plain text of the whole PDF, pages joined the same way as zarai.kb.ingest.extract_text."""
    return PAGE_SEPARATOR.join(p.text for p in ocr_pdf_pages(path, lang=lang, dpi=dpi))


def write_report(pages: list[PageResult], path: str | Path) -> None:
    """One row per page so low-confidence pages can be checked by hand."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["page", "mean_conf", "n_words", "urdu_share", "n_chars"])
        for p in pages:
            writer.writerow(
                [
                    p.page,
                    round(p.mean_conf, 1),
                    p.n_words,
                    round(urdu_share(p.text), 3),
                    len(p.text),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True, help="PDF file")
    parser.add_argument("--out", required=True, help="output .txt")
    parser.add_argument("--report", help="optional per-page CSV report")
    parser.add_argument("--lang", default="urd+eng")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--psm", type=int, default=4)
    args = parser.parse_args()

    log = get_logger(__name__)
    pages = ocr_pdf_pages(args.input, lang=args.lang, dpi=args.dpi, psm=args.psm)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(PAGE_SEPARATOR.join(p.text for p in pages), encoding="utf-8")
    if args.report:
        write_report(pages, args.report)
    confs = [p.mean_conf for p in pages if p.mean_conf >= 0]
    low = [p.page for p in pages if 0 <= p.mean_conf < 60]
    log.info(
        "%s: %d pages, mean confidence %.1f, pages under 60: %s",
        args.input,
        len(pages),
        sum(confs) / len(confs) if confs else -1,
        low or "none",
    )


if __name__ == "__main__":
    main()
