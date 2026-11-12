"""Embedding models + an on-disk cache.

Embeddings are computed once with OpenAI ``text-embedding-3-small`` and saved in
``data/embeddings_cache.npz`` (committed to the repo). Benchmarks and tests then
run offline, deterministically and for free; only new live-search queries call
the API.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data" / "embeddings_cache.npz"


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale every row to unit length, so dot product == cosine similarity."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def text_key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\x00{text}".encode("utf-8")).hexdigest()


class OpenAIEmbedder:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None, batch_size: int = 96):
        from openai import OpenAI

        key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        self.client = OpenAI(api_key=key)
        self.model_name = model
        self.batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            resp = self.client.embeddings.create(model=self.model_name, input=list(texts[i : i + self.batch_size]))
            vectors.extend(d.embedding for d in resp.data)
        return l2_normalize(np.array(vectors, dtype=np.float32))


class HashingEmbedder:
    """Offline bag-of-words hashing embedder.

    NOT semantic. Used only to unit-test index mechanics (normalization,
    distance-to-similarity, ranking) without an API key.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.model_name = f"hashing-{dim}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in re.findall(r"[a-z0-9_]+", text.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[row, h % self.dim] += 1.0 if (h >> 64) % 2 else -1.0
        return l2_normalize(out)


class MissingEmbeddingError(RuntimeError):
    pass


class CachedEmbedder:
    """Looks up vectors in the cache; calls ``fallback`` only for misses."""

    def __init__(self, cache_path: Path | str = DEFAULT_CACHE, model: str = DEFAULT_MODEL, fallback: Embedder | None = None):
        self.cache_path = Path(cache_path)
        self.model_name = model
        self.fallback = fallback
        self._vectors: dict[str, np.ndarray] = {}
        if self.cache_path.exists():
            data = np.load(self.cache_path, allow_pickle=False)
            for k, v in zip(data["keys"], data["vectors"]):
                self._vectors[str(k)] = v.astype(np.float32)
        self.new_entries = 0

    def __contains__(self, text: str) -> bool:
        return text_key(self.model_name, text) in self._vectors

    def missing(self, texts: Sequence[str]) -> list[str]:
        return [t for t in dict.fromkeys(texts) if t not in self]

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        todo = self.missing(texts)
        if todo:
            if self.fallback is None:
                raise MissingEmbeddingError(
                    f"{len(todo)} text(s) are not in {self.cache_path.name}. "
                    "Run `uv run python scripts/build_embeddings.py` with OPENAI_API_KEY set."
                )
            for t, v in zip(todo, self.fallback.embed(todo)):
                self._vectors[text_key(self.model_name, t)] = v
            self.new_entries += len(todo)
        return l2_normalize(np.stack([self._vectors[text_key(self.model_name, t)] for t in texts]))

    def save(self) -> None:
        keys = np.array(list(self._vectors.keys()))
        vecs = np.stack(list(self._vectors.values())).astype(np.float16)  # half the size; ample precision
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.cache_path, keys=keys, vectors=vecs)
