"""Embed every chunk + every evaluation/demo query ONCE and save the cache.

Run (needs OPENAI_API_KEY in .env, costs well under $0.01):
    uv run python scripts/build_embeddings.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from src.benchmarker import load_eval  # noqa: E402
from src.embeddings import CachedEmbedder, OpenAIEmbedder  # noqa: E402
from src.search import SearchEngine  # noqa: E402
from src.demo_queries import DEMO_QUERIES  # noqa: E402


def main() -> None:
    engine = SearchEngine.from_corpus()
    texts = [c.content for chunks in engine.chunk_sets.values() for c in chunks]
    texts += [q.query for q in load_eval()] + DEMO_QUERIES
    cache = CachedEmbedder(fallback=OpenAIEmbedder())
    todo = cache.missing(texts)
    print(f"{len(texts)} texts, {len(todo)} not cached yet")
    if todo:
        cache.embed(todo)
        cache.save()
    print(f"Saved {cache.cache_path.relative_to(ROOT)} ({cache.new_entries} new embeddings)")


if __name__ == "__main__":
    main()
