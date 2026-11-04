"""Convert every source in kb/source_register.csv (pdf/txt/html) to plain text."""

import logging
from pathlib import Path

from zarai.common.config import parse_config
from zarai.common.io import read_csv
from zarai.common.logging import get_logger

log = logging.getLogger(__name__)


def extract_text(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        from pypdf import PdfReader

        return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    if suffix in {".html", ".htm"}:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text("\n")
    raise ValueError(f"unsupported file type: {path}")


def main() -> None:
    cfg = parse_config(__doc__)
    get_logger(__name__)
    text_dir = Path(cfg.text_dir)
    text_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(cfg.register)
    for row in rows:
        text = extract_text(row["file_path"])
        if not text.strip():
            log.warning("%s: no text extracted, probably a scan that needs OCR", row["source_id"])
            continue
        (text_dir / f"{row['source_id']}.txt").write_text(text, encoding="utf-8")
    log.info("processed %d sources into %s", len(rows), text_dir)


if __name__ == "__main__":
    main()
