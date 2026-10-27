"""Multi-format document loaders.

Each loader turns one file into a list of :class:`Section` objects (heading + text, plus the
page number for PDFs). :func:`load_document` wraps that into a :class:`Document` with
metadata, and :func:`load_corpus` walks a directory tree.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from pathlib import Path

from bs4 import BeautifulSoup, Tag
from pypdf import PdfReader

from hyrag.ingestion.normalize import normalize_text, strip_markdown_inline
from hyrag.models import DocFormat, Document, Section, stable_hash

SectionLoader = Callable[[Path], tuple[str | None, list[Section]]]

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_MD_FENCE = re.compile(r"^(```|~~~)")
_SETEXT_UNDERLINE = re.compile(r"^(=+|-+)\s*$")
# "2. Impact" or "2.1 Detection"; a bare number ("18 replicas ...") is not a heading.
_PDF_HEADING = re.compile(r"^(?:\d+\.|\d+(?:\.\d+)+\.?)\s+[A-Z][^.:;]{1,70}$")


class _SectionBuilder:
    """Tracks the heading hierarchy and collects text under the current heading."""

    def __init__(self) -> None:
        self.sections: list[Section] = []
        self._path: list[tuple[int, str]] = []
        self._buffer: list[str] = []
        self._page: int | None = None

    @property
    def heading(self) -> str | None:
        return " > ".join(title for _, title in self._path) or None

    def add_text(self, text: str) -> None:
        if text.strip():
            self._buffer.append(text)

    def start_section(self, level: int, title: str) -> None:
        self.flush()
        self._path = [(lvl, t) for lvl, t in self._path if lvl < level]
        self._path.append((level, title.strip()))

    def start_page(self, page: int) -> None:
        self.flush()
        self._page = page

    def flush(self) -> None:
        text = normalize_text("\n".join(self._buffer))
        if text:
            level = self._path[-1][0] if self._path else 0
            self.sections.append(
                Section(heading=self.heading, text=text, page=self._page, level=level)
            )
        self._buffer = []


def load_markdown(path: Path) -> tuple[str | None, list[Section]]:
    builder = _SectionBuilder()
    title: str | None = None
    in_fence = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if _MD_FENCE.match(line.strip()):
            in_fence = not in_fence
            continue
        match = None if in_fence else _MD_HEADING.match(line)
        if match:
            level, heading = len(match.group(1)), strip_markdown_inline(match.group(2))
            title = title or (heading if level == 1 else None)
            builder.start_section(level, heading)
        else:
            builder.add_text(line if in_fence else strip_markdown_inline(line))
    builder.flush()
    return title, builder.sections


def load_text(path: Path) -> tuple[str | None, list[Section]]:
    """Plain text; setext-style underlined lines (=== or ---) are treated as headings."""
    builder = _SectionBuilder()
    title: str | None = None
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if line.strip() and _SETEXT_UNDERLINE.match(nxt):
            level = 1 if nxt.strip().startswith("=") else 2
            title = title or (line.strip() if level == 1 else None)
            builder.start_section(level, line.strip())
            i += 2
            continue
        builder.add_text(line)
        i += 1
    builder.flush()
    return title, builder.sections


_HTML_HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4}
_HTML_BLOCKS = ("p", "li", "pre", "table", "dt", "dd", "blockquote")


def _table_text(table: Tag) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def load_html(path: Path) -> tuple[str | None, list[Section]]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else None

    builder = _SectionBuilder()
    root = soup.body or soup
    for el in root.find_all([*_HTML_HEADINGS, *_HTML_BLOCKS]):
        # Skip blocks nested inside another captured block (e.g. <p> inside <li>).
        if el.find_parent(list(_HTML_BLOCKS)):
            continue
        if el.name in _HTML_HEADINGS:
            heading = el.get_text(" ", strip=True)
            title = title or heading
            builder.start_section(_HTML_HEADINGS[el.name], heading)
        elif el.name == "table":
            builder.add_text(_table_text(el))
        elif el.name == "li":
            builder.add_text("- " + el.get_text(" ", strip=True))
        elif el.name == "pre":
            builder.add_text(el.get_text())
        else:
            builder.add_text(el.get_text(" ", strip=True))
    builder.flush()
    return title, builder.sections


def load_pdf(path: Path) -> tuple[str | None, list[Section]]:
    """One section per (heading, page) pair so every chunk can cite an exact page.

    Headings are detected with a numbering heuristic ("2.1 Rollback steps"), which matches
    how most internal PDFs (postmortems, policies) are laid out.
    """
    reader = PdfReader(str(path))
    meta_title = reader.metadata.title if reader.metadata and reader.metadata.title else None
    title: str | None = str(meta_title) if meta_title else None
    builder = _SectionBuilder()
    for page_no, page in enumerate(reader.pages, start=1):
        builder.start_page(page_no)
        for raw_line in (page.extract_text() or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if title is None:
                title = line
                continue
            if _PDF_HEADING.match(line):
                level = line.split()[0].rstrip(".").count(".") + 1
                builder.start_section(level, line)
            else:
                builder.add_text(line)
    builder.flush()
    return title, builder.sections


LOADERS: dict[str, tuple[DocFormat, SectionLoader]] = {
    ".md": ("markdown", load_markdown),
    ".markdown": ("markdown", load_markdown),
    ".txt": ("text", load_text),
    ".html": ("html", load_html),
    ".htm": ("html", load_html),
    ".pdf": ("pdf", load_pdf),
}

SUPPORTED_EXTENSIONS = frozenset(LOADERS)


class UnsupportedFormatError(ValueError):
    pass


def load_document(path: Path, root: Path | None = None) -> Document:
    """Load one file. ``source`` is recorded relative to ``root`` (defaults to the parent)."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in LOADERS:
        raise UnsupportedFormatError(f"Unsupported file type: {path.name}")
    fmt, loader = LOADERS[ext]
    root = Path(root) if root is not None else path.parent
    source = path.resolve().relative_to(root.resolve()).as_posix()

    title, sections = loader(path)
    return Document(
        doc_id=stable_hash(source, length=12),
        source=source,
        title=title or path.stem.replace("-", " ").replace("_", " ").title(),
        format=fmt,
        sections=sections,
        doc_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
        metadata={"filename": path.name, "extension": ext},
    )


def iter_corpus_files(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in LOADERS)


def load_corpus(root: Path) -> list[Document]:
    """Load every supported file under ``root``; documents with no text are dropped."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {root}")
    docs = [load_document(p, root) for p in iter_corpus_files(root)]
    return [d for d in docs if d.sections]
