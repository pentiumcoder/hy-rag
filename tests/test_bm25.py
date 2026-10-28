from __future__ import annotations

from pathlib import Path

from hyrag.index.bm25_store import BM25Store
from hyrag.index.tokenize import tokenize


def test_tokenize_keeps_identifiers_whole_and_emits_parts() -> None:
    tokens = tokenize("Set ingest.max_batch_size above 5000 and you get NW-4012.")
    assert "ingest.max_batch_size" in tokens
    assert {"ingest", "max", "batch", "size"} <= set(tokens)
    assert {"nw-4012", "nw", "4012", "5000"} <= set(tokens)
    assert "and" not in tokens and "the" not in tokens


def test_tokenize_headers_paths_and_edge_cases() -> None:
    assert tokenize("X-NW-Signature")[:1] == ["x-nw-signature"]
    assert "v2/events" in tokenize("POST /v2/events")
    assert tokenize("the of and") == []
    assert tokenize("__init__") == ["init"]
    assert tokenize("") == []


def _store() -> BM25Store:
    store = BM25Store()
    store.add(
        [
            ("cfg", "ingest.max_batch_size: maximum events per batch. Default 500, max 5000."),
            ("err", "NW-4012 BatchTooLarge: the batch exceeds ingest.max_batch_size."),
            ("pto", "Employees accrue 25 days of paid time off per year."),
            ("rate", "Rate limits: Starter 50 rps, Growth 100 rps, Enterprise 1000 rps."),
        ]
    )
    return store


def test_search_ranks_exact_identifier_matches() -> None:
    hits = _store().search("What does ingest.max_batch_size control?", k=10)
    assert {cid for cid, _ in hits[:2]} == {"cfg", "err"}
    assert all(score > 0 for _, score in hits)
    assert "pto" not in {cid for cid, _ in hits}  # no shared term -> never returned


def test_search_error_code_and_limits() -> None:
    store = _store()
    assert store.search("NW-4012", k=1)[0][0] == "err"
    assert store.search("enterprise rate limits", k=1)[0][0] == "rate"
    assert store.search("the and of", k=5) == []
    assert BM25Store().search("anything") == []


def test_add_remove_and_persist(tmp_path: Path) -> None:
    path = tmp_path / "bm25.json"
    store = _store()
    store.path = path
    store.remove(["pto", "missing"])
    store.add([("cfg", "replaced text about webhooks")])
    store.save()

    loaded = BM25Store.load(path)
    assert loaded.ids() == {"cfg", "err", "rate"}
    assert len(loaded) == 3
    assert loaded.search("webhooks", k=1)[0][0] == "cfg"
    assert loaded.search("NW-4012", k=1) == store.search("NW-4012", k=1)
    assert not path.with_suffix(".tmp").exists()

    loaded.clear()
    assert loaded.ids() == set()
    assert BM25Store.load(tmp_path / "missing.json").ids() == set()
    BM25Store().save()  # in-memory store: save is a no-op


def test_common_terms_keep_a_positive_weight() -> None:
    """Okapi idf is 0 for a term in half the chunks; the Lucene variant stays positive."""
    store = BM25Store()
    store.add([("a", "batch events"), ("b", "batch size"), ("c", "pto"), ("d", "leave")])
    hits = store.search("batch", k=5)
    assert {cid for cid, _ in hits} == {"a", "b"}
    assert all(score > 0 for _, score in hits)
