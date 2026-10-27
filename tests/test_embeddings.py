from __future__ import annotations

import numpy as np
import pytest

from hyrag.embeddings import Embedder, FakeEmbedder, cosine_similarity_matrix, get_embedder


def test_fake_embedder_is_normalized_and_deterministic() -> None:
    emb = FakeEmbedder(dim=64)
    a = emb.embed_documents(["rate limit exceeded", "rate limit exceeded"])
    assert a.shape == (2, 64)
    assert a.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, rtol=1e-5)
    np.testing.assert_array_equal(a[0], a[1])
    np.testing.assert_array_equal(FakeEmbedder(dim=64).embed_query("rate limit exceeded"), a[0])


def test_fake_embedder_similarity_tracks_vocabulary_overlap() -> None:
    emb = FakeEmbedder()
    vecs = emb.embed_documents(
        [
            "ingest batch size limit for events",
            "the maximum batch size for ingest events",
            "paid time off carryover policy",
        ]
    )
    sims = cosine_similarity_matrix(vecs, vecs)
    assert sims[0, 1] > 0.3
    assert sims[0, 1] > sims[0, 2] + 0.2


def test_fake_embedder_handles_empty_input() -> None:
    emb = FakeEmbedder(dim=8)
    assert emb.embed_documents([]).shape == (0, 8)
    assert np.linalg.norm(emb.embed_query("")) == 0.0  # no tokens -> zero vector, no NaN


def test_get_embedder() -> None:
    assert isinstance(get_embedder("fake"), Embedder)
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        get_embedder("nope")
