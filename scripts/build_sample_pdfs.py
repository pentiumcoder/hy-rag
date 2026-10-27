"""Render the PDF part of the Northwind sample corpus from plain-text sources.

Source format (scripts/pdf_sources/*.txt): first line is the title, numbered lines such as
"2. Impact" or "2.1 Rotation" are headings, blank lines separate paragraphs, and lines
starting with "- " are bullets. Output is deterministic (fixed creation date), so
rebuilding does not change the committed PDFs unless a source changed.

    python scripts/build_sample_pdfs.py
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "scripts" / "pdf_sources"
CORPUS = ROOT / "data" / "corpus"

OUTPUTS = {
    "2026-03-ingest-outage.txt": "postmortems/2026-03-ingest-outage.pdf",
    "2026-06-auth-token-incident.txt": "postmortems/2026-06-auth-token-incident.pdf",
    "security-policy.txt": "policies/security-policy.pdf",
    "travel-and-expense.txt": "finance/travel-and-expense.pdf",
}

HEADING = re.compile(r"^(?:\d+\.|\d+(?:\.\d+)+\.?)\s+[A-Z]")
FIXED_DATE = datetime(2026, 7, 1, tzinfo=UTC)


def render(source: Path, target: Path) -> None:
    lines = source.read_text(encoding="utf-8").splitlines()
    title, body = lines[0], lines[1:]

    pdf = FPDF(format="A4")
    pdf.set_creation_date(FIXED_DATE)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(20, 20, 20)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 9, title, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    for line in body:
        if not line.strip():
            pdf.ln(2)
        elif HEADING.match(line):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 7, line, new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.set_font("Helvetica", "", 10.5)
            pdf.multi_cell(0, 5.5, line, new_x="LMARGIN", new_y="NEXT")

    target.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(target))


def main() -> None:
    for name, rel in OUTPUTS.items():
        render(SOURCES / name, CORPUS / rel)
        print(f"wrote {rel}")


if __name__ == "__main__":
    main()
