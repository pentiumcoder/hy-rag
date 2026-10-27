from hyrag.ingestion.normalize import normalize_text, strip_markdown_inline


def test_normalize_collapses_whitespace_and_newlines() -> None:
    raw = "Hello   world\r\n\r\n\r\n\r\nNext\tline  "
    assert normalize_text(raw) == "Hello world\n\nNext line"


def test_normalize_rejoins_pdf_hyphenation_only_for_lowercase_words() -> None:
    assert normalize_text("inges-\ntion pipeline") == "ingestion pipeline"
    assert normalize_text("nw-\nGateway") == "nw-\nGateway"


def test_normalize_unicode_and_nul() -> None:
    assert normalize_text("ﬁle\x00 name") == "file name"


def test_strip_markdown_keeps_identifiers() -> None:
    text = "Set **`ingest.max_batch_size`** per the [config docs](config.md). ![img](a.png)"
    assert strip_markdown_inline(text) == "Set ingest.max_batch_size per the config docs. img"
