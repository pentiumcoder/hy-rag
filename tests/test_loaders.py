from __future__ import annotations

from pathlib import Path

import pytest

from hyrag.ingestion import UnsupportedFormatError, load_corpus, load_document
from tests.conftest import write_pdf


def test_markdown_sections_follow_heading_hierarchy(tmp_path: Path) -> None:
    md = tmp_path / "guide.md"
    md.write_text(
        "# Guide\nIntro text.\n\n## Setup\nInstall `nwctl`.\n\n### Linux\nUse apt.\n\n"
        "## Usage\n```\n# not a heading\nnwctl doctor\n```\n",
        encoding="utf-8",
    )
    doc = load_document(md)
    assert doc.title == "Guide"
    assert doc.format == "markdown"
    assert [s.heading for s in doc.sections] == [
        "Guide",
        "Guide > Setup",
        "Guide > Setup > Linux",
        "Guide > Usage",
    ]
    assert doc.sections[1].text == "Install nwctl."
    assert "# not a heading" in doc.sections[3].text  # fenced code kept verbatim
    assert doc.sections[2].level == 3


def test_text_setext_headings(tmp_path: Path) -> None:
    txt = tmp_path / "faq.txt"
    txt.write_text("Leave FAQ\n=========\n\nIntro.\n\nQuestion one\n------------\nAnswer.\n")
    doc = load_document(txt)
    assert doc.title == "Leave FAQ"
    assert [s.heading for s in doc.sections] == ["Leave FAQ", "Leave FAQ > Question one"]
    assert doc.sections[1].text == "Answer."


def test_text_without_headings_falls_back_to_filename_title(tmp_path: Path) -> None:
    txt = tmp_path / "plain_notes.txt"
    txt.write_text("just some text\nmore text\n")
    doc = load_document(txt)
    assert doc.title == "Plain Notes"
    assert doc.sections[0].heading is None


def test_html_strips_chrome_and_reads_tables(tmp_path: Path) -> None:
    html = tmp_path / "page.html"
    html.write_text(
        "<html><head><title>Page</title><script>var x=1;</script></head><body>"
        "<nav>menu</nav><h1>Main</h1><p>First para.</p>"
        "<h2>Limits</h2><table><tr><th>Plan</th><th>RPS</th></tr>"
        "<tr><td>Starter</td><td>50</td></tr></table>"
        "<ul><li>item <p>nested</p></li></ul><pre>nwctl doctor</pre>"
        "<footer>owned by x</footer></body></html>",
        encoding="utf-8",
    )
    doc = load_document(html)
    assert doc.title == "Page"
    text = doc.text
    assert "menu" not in text and "var x" not in text and "owned by" not in text
    limits = doc.sections[1]
    assert limits.heading == "Main > Limits"
    assert "Plan | RPS\nStarter | 50" in limits.text
    assert "- item nested" in limits.text
    assert text.count("nested") == 1
    assert "nwctl doctor" in limits.text


def test_pdf_sections_carry_page_numbers(tmp_path: Path) -> None:
    pdf = write_pdf(
        tmp_path / "pm.pdf",
        "Postmortem: Test",
        ["1. Summary", "Things broke.", "2. Root cause", "A config value.", "<page>", "More."],
    )
    doc = load_document(pdf)
    assert doc.title == "Postmortem: Test"
    assert doc.format == "pdf"
    assert [(s.heading, s.page) for s in doc.sections] == [
        ("1. Summary", 1),
        ("2. Root cause", 1),
        ("2. Root cause", 2),
    ]
    assert doc.sections[2].text == "More."


def test_source_is_relative_posix_and_ids_are_stable(tmp_path: Path) -> None:
    (tmp_path / "a" / "b").mkdir(parents=True)
    f = tmp_path / "a" / "b" / "x.md"
    f.write_text("# X\nbody\n")
    d1, d2 = load_document(f, tmp_path), load_document(f, tmp_path)
    assert d1.source == "a/b/x.md"
    assert d1.doc_id == d2.doc_id and d1.doc_hash == d2.doc_hash
    f.write_text("# X\nchanged\n")
    assert load_document(f, tmp_path).doc_hash != d1.doc_hash


def test_unsupported_format_raises(tmp_path: Path) -> None:
    f = tmp_path / "x.docx"
    f.write_bytes(b"binary")
    with pytest.raises(UnsupportedFormatError):
        load_document(f)


def test_load_corpus_skips_unsupported_and_empty(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\ntext\n")
    (tmp_path / "empty.txt").write_text("   \n")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    docs = load_corpus(tmp_path)
    assert [d.source for d in docs] == ["a.md"]


def test_load_corpus_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "nope")
