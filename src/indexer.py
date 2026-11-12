"""Dense (ChromaDB, cosine) and sparse (BM25) indexes over the same chunks."""

from __future__ import annotations

import re
import uuid
from typing import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from src.chunking import DocumentChunk
from src.embeddings import Embedder, l2_normalize

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def tokenize(text: str) -> list[str]:
    """The ONE tokenizer used for BM25 documents AND queries.

    Lower-cases, strips punctuation, keeps identifiers whole
    (``fst_err_hook_timeout``, ``maxparamlength``) and also adds their parts
    (``max``, ``param``, ``length``) so ``maxParamLength`` matches "max param length".
    """
    out: list[str] = []
    for word in _WORD_RE.findall(text):
        out.append(word.lower())
        parts = [p.lower() for piece in word.split("_") for p in _CAMEL_RE.findall(piece)]
        if len(parts) > 1:
            out.extend(parts)
    return out


class BM25Index:
    def __init__(self, chunks: Sequence[DocumentChunk]):
        self.chunks = list(chunks)
        self.bm25 = BM25Okapi([tokenize(c.content) for c in self.chunks])

    def search(self, query: str, k: int) -> list[tuple[DocumentChunk, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]


class DenseIndex:
    """ChromaDB collection using cosine space with pre-normalized embeddings."""

    def __init__(self, chunks: Sequence[DocumentChunk], embedder: Embedder):
        import chromadb
        from chromadb.config import Settings

        self.chunks = {c.chunk_id: c for c in chunks}
        self.embedder = embedder
        client = chromadb.EphemeralClient(Settings(anonymized_telemetry=False, allow_reset=True))
        self.collection = client.create_collection(
            name=f"chunks-{uuid.uuid4().hex[:12]}", metadata={"hnsw:space": "cosine"}
        )
        vectors = l2_normalize(embedder.embed([c.content for c in chunks]))
        ids = [c.chunk_id for c in chunks]
        for i in range(0, len(ids), 4000):  # Chroma max batch size
            self.collection.add(
                ids=ids[i : i + 4000],
                embeddings=vectors[i : i + 4000].tolist(),
                metadatas=[{"parent_doc": c.parent_doc, "header_path": c.header_path} for c in chunks[i : i + 4000]],
            )

    def search(self, query: str, k: int, query_vector: np.ndarray | None = None) -> list[tuple[DocumentChunk, float]]:
        vec = l2_normalize(self.embedder.embed([query]))[0] if query_vector is None else query_vector
        res = self.collection.query(query_embeddings=[vec.tolist()], n_results=min(k, len(self.chunks)))
        # Chroma returns cosine DISTANCE (0 = identical). Convert to similarity
        # so that a higher score always means "more relevant".
        return [
            (self.chunks[cid], 1.0 - float(dist))
            for cid, dist in zip(res["ids"][0], res["distances"][0])
        ]
