"""Unified search interface over every chunking strategy and both index types."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from src.chunking import DocumentChunk, FixedSizeChunker, StructureAwareChunker, chunk_corpus, load_corpus
from src.embeddings import Embedder
from src.indexer import BM25Index, DenseIndex, tokenize

Method = Literal["bm25", "dense"]
ROOT = Path(__file__).resolve().parent.parent

# name -> chunker. Overlap is 12.5% of the window.
CHUNKERS = {
    "fixed-256": lambda: FixedSizeChunker(256, 32),
    "fixed-512": lambda: FixedSizeChunker(512, 64),
    "fixed-1024": lambda: FixedSizeChunker(1024, 128),
    "structure": lambda: StructureAwareChunker(max_tokens=512),
}

# The configurations compared in the benchmark: (label, method, chunking)
BENCHMARK_CONFIGS: list[tuple[str, Method, str]] = [
    ("BM25 (baseline, structure chunks)", "bm25", "structure"),
    ("Dense · fixed 256", "dense", "fixed-256"),
    ("Dense · fixed 512", "dense", "fixed-512"),
    ("Dense · fixed 1024", "dense", "fixed-1024"),
    ("Dense · structure-aware", "dense", "structure"),
]


class SearchResult(BaseModel):
    rank: int
    chunk_id: str
    score: float
    score_type: Literal["bm25", "cosine_similarity"]
    parent_doc: str
    header_path: str
    content: str
    char_start: int
    char_end: int
    matched_terms: list[str]


@dataclass
class SearchResponse:
    results: list[SearchResult]
    latency_ms: float


@dataclass
class SearchEngine:
    docs: dict[str, str]
    embedder: Embedder | None
    chunk_sets: dict[str, list[DocumentChunk]] = field(default_factory=dict)
    _bm25: dict[str, BM25Index] = field(default_factory=dict)
    _dense: dict[str, DenseIndex] = field(default_factory=dict)

    @classmethod
    def from_corpus(cls, corpus_dir: str | Path = ROOT / "corpus", embedder: Embedder | None = None, strategies=None) -> "SearchEngine":
        eng = cls(docs=load_corpus(corpus_dir), embedder=embedder)
        for name in strategies or CHUNKERS:
            eng.chunk_sets[name] = chunk_corpus(eng.docs, CHUNKERS[name]())
        return eng

    def bm25(self, strategy: str) -> BM25Index:
        if strategy not in self._bm25:
            self._bm25[strategy] = BM25Index(self.chunk_sets[strategy])
        return self._bm25[strategy]

    def dense(self, strategy: str) -> DenseIndex:
        if strategy not in self._dense:
            if self.embedder is None:
                raise RuntimeError("Dense search needs an embedder")
            self._dense[strategy] = DenseIndex(self.chunk_sets[strategy], self.embedder)
        return self._dense[strategy]

    def search(self, query: str, method: Method, strategy: str, k: int = 5) -> SearchResponse:
        index = self.bm25(strategy) if method == "bm25" else self.dense(strategy)  # build outside the timer
        q_terms = set(tokenize(query))
        t0 = time.perf_counter()
        hits = index.search(query, k)
        latency = (time.perf_counter() - t0) * 1000
        results = [
            SearchResult(
                rank=i + 1,
                chunk_id=c.chunk_id,
                score=round(s, 4),
                score_type="bm25" if method == "bm25" else "cosine_similarity",
                parent_doc=c.parent_doc,
                header_path=c.header_path,
                content=c.content,
                char_start=c.metadata["char_start"],
                char_end=c.metadata["char_end"],
                matched_terms=sorted(q_terms & set(tokenize(c.content))),
            )
            for i, (c, s) in enumerate(hits)
        ]
        return SearchResponse(results=results, latency_ms=latency)
