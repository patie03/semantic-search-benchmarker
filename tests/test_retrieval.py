"""Deterministic retrieval tests for BM25 and dense search."""

import numpy as np
import pytest

from src.benchmarker import keyword_overlap, load_eval, relevant_ids
from src.chunking import StructureAwareChunker
from src.embeddings import HashingEmbedder, l2_normalize
from src.indexer import BM25Index, DenseIndex, tokenize
from tests.conftest import FIXTURE_DOC


# Test 3 --------------------------------------------------------------------- #
def test_rare_identifier_is_rank_1_in_bm25_fixture():
    chunks = StructureAwareChunker().split(FIXTURE_DOC, "fixture.md")
    top = BM25Index(chunks).search("CUSTOM_CONFIG_FLAG_XYZ", k=3)
    assert "cache warmer" in top[0][0].content
    assert top[0][1] > top[1][1]


def test_real_error_code_is_rank_1_in_bm25(engine):
    q = next(q for q in load_eval() if q.id == "lex_hook_timeout")
    rel = relevant_ids(engine.chunk_sets["structure"], engine.docs[q.doc], q)
    top = engine.search(q.query, "bm25", "structure", k=5).results
    assert top[0].chunk_id in rel


def test_bm25_query_and_document_tokenization_are_symmetric():
    # Same tokenizer for both sides: casing, punctuation and camelCase don't matter.
    assert tokenize("maxParamLength") == tokenize("MAXPARAMLENGTH")[:1] + tokenize("maxParamLength")[1:]
    assert "maxparamlength" in tokenize("`maxParamLength`,") and "param" in tokenize("maxParamLength")
    assert tokenize("Error: FST_ERR_HOOK_TIMEOUT!")[:2] == ["error", "fst_err_hook_timeout"]


# Test 4 --------------------------------------------------------------------- #
def test_paraphrase_with_zero_shared_keywords_found_by_dense(engine, has_embeddings):
    if not has_embeddings:
        pytest.skip("run scripts/build_embeddings.py first")
    q = next(q for q in load_eval() if q.id == "sem_bodylimit")
    chunks = engine.chunk_sets["structure"]
    rel = relevant_ids(chunks, engine.docs[q.doc], q)
    # the query really shares no keyword with the answer chunk ...
    assert all(keyword_overlap(q.query, c.content) == [] for c in chunks if c.chunk_id in rel)
    # ... so BM25 can't find it, but dense search can
    dense = [r.chunk_id for r in engine.search(q.query, "dense", "structure", k=5).results]
    assert set(dense) & rel


# Index mechanics ------------------------------------------------------------ #
def test_dense_scores_are_similarities_highest_first():
    chunks = StructureAwareChunker().split(FIXTURE_DOC, "fixture.md")
    index = DenseIndex(chunks, HashingEmbedder())
    target = next(c for c in chunks if "handshake" in c.content)
    hits = index.search(target.content, k=len(chunks))
    assert hits[0][0].chunk_id == target.chunk_id  # identical text ranks first
    assert hits[0][1] == pytest.approx(1.0, abs=1e-4)  # similarity, not distance (would be 0)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_embeddings_are_unit_length():
    vecs = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(np.linalg.norm(vecs[0]), 1.0) and np.allclose(vecs[1], 0.0)
