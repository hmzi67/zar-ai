import pytest

from zarai.kb import ocr


def test_urdu_share_ignores_digits_and_punctuation():
    assert ocr.urdu_share("گندم 2023-24") == 1.0
    assert ocr.urdu_share("2023 15 21-") == 0.0
    assert ocr.urdu_share("گندم wheat") == pytest.approx(4 / 9)


def test_digits_only_urdu_pdf_needs_ocr():
    # What pypdf returns for the AARI wheat plan: numbers and a few English words.
    broken = "2829 28.62 18377 6496 16052 2018-19\n" * 20 + "Climate change Rotavator SSP"
    assert ocr.needs_ocr(broken, "ur")


def test_good_urdu_text_does_not_need_ocr():
    assert not ocr.needs_ocr("گندم کی کاشت کا بہترین وقت نومبر ہے۔ " * 20, "ur")


def test_english_source_is_not_judged_by_urdu_share():
    assert not ocr.needs_ocr("Wheat sowing should finish by the end of November. " * 10, "en")


def test_short_text_needs_ocr():
    assert ocr.needs_ocr("", "en")


def test_pages_are_joined_like_ingest(monkeypatch):
    pages = [ocr.PageResult(1, "پہلا", 90.0, 1), ocr.PageResult(2, "دوسرا", 80.0, 1)]
    monkeypatch.setattr(ocr, "ocr_pdf_pages", lambda *a, **k: pages)
    assert ocr.ocr_pdf("x.pdf") == "پہلا\n\nدوسرا"


def test_report_has_one_row_per_page(tmp_path):
    pages = [ocr.PageResult(1, "گندم", 91.2, 1), ocr.PageResult(2, "", -1.0, 0)]
    out = tmp_path / "r.csv"
    ocr.write_report(pages, out)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "page,mean_conf,n_words,urdu_share,n_chars"
    assert len(lines) == 3
