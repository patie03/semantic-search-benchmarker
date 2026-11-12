from pathlib import Path

import pytest

from src.embeddings import DEFAULT_CACHE, CachedEmbedder
from src.search import SearchEngine

FIXTURE_DOC = """# Config Reference

## Overview

This service reads settings at boot and validates them.

## Flags

### CUSTOM_CONFIG_FLAG_XYZ

Enables the experimental cache warmer. Default: `false`.

```js
const app = build({
  CUSTOM_CONFIG_FLAG_XYZ: true,
  retries: 3
})
```

### retries

How many times a failed request is attempted again before giving up.

## Networking

Sockets are pooled and reused across requests to reduce handshake cost.
"""


@pytest.fixture(scope="session")
def engine() -> SearchEngine:
    """Real Fastify corpus, all chunking strategies, cached OpenAI embeddings."""
    return SearchEngine.from_corpus(embedder=CachedEmbedder())


@pytest.fixture(scope="session")
def has_embeddings() -> bool:
    return Path(DEFAULT_CACHE).exists()
