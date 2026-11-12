"""Information-retrieval benchmark: Hit Rate@K, Recall@K, MRR@K and latency.

Ground truth is defined by a SOURCE TEXT SPAN (file + exact answer text), not
by a chunk id. For every chunking strategy, a chunk is relevant if its source
character range overlaps the answer span. This keeps the labels valid when the
chunk size changes (no "chunk #12" index leakage).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel

from src.chunking import DocumentChunk
from src.indexer import tokenize
from src.search import BENCHMARK_CONFIGS, SearchEngine

ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = ROOT / "data" / "eval_queries.json"
STOPWORDS = set(
    "a an the of to in on for and or is are be by with how do does i my can what which when why it this that from "
    "you your if will at as not into out get".split()
)


class EvalQuery(BaseModel):
    id: str
    query: str
    type: Literal["lexical", "semantic"]
    doc: str
    section: str
    answer_span: str


# --------------------------------------------------------------------------- #
# Metric math (pure functions, unit-tested)
# --------------------------------------------------------------------------- #
def _dedupe(ids: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(ids))


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant id is in the top-k, else 0.0."""
    return 1.0 if relevant and any(r in relevant for r in _dedupe(ranked)[:k]) else 0.0


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Share of relevant ids found in the top-k (duplicates counted once)."""
    if not relevant:
        return 0.0
    return len(set(_dedupe(ranked)[:k]) & relevant) / len(relevant)


def reciprocal_rank_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """1/rank of the first relevant id within the top-k; 0.0 if none (never divides by zero)."""
    for i, r in enumerate(_dedupe(ranked)[:k], start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #
def load_eval(path: str | Path = EVAL_PATH) -> list[EvalQuery]:
    return [EvalQuery(**q) for q in json.loads(Path(path).read_text(encoding="utf-8"))]


def locate_span(doc_text: str, span: str) -> tuple[int, int]:
    """Find ``span`` in the document, ignoring differences in whitespace/line breaks."""
    pattern = r"\s+".join(re.escape(w) for w in span.split())
    m = re.search(pattern, doc_text)
    if not m:
        raise ValueError(f"answer span not found: {span[:60]!r}")
    return m.start(), m.end()


def relevant_ids(chunks: Sequence[DocumentChunk], doc_text: str, q: EvalQuery) -> set[str]:
    s0, s1 = locate_span(doc_text, q.answer_span)
    return {
        c.chunk_id
        for c in chunks
        if c.parent_doc == q.doc and c.metadata["char_start"] < s1 and c.metadata["char_end"] > s0
    }


def keyword_overlap(query: str, text: str) -> list[str]:
    return sorted((set(tokenize(query)) - STOPWORDS) & set(tokenize(text)))


# --------------------------------------------------------------------------- #
# Benchmark runner
# --------------------------------------------------------------------------- #
def run_benchmark(engine: SearchEngine, queries: Sequence[EvalQuery], ks: Sequence[int] = (3, 5, 10), configs=BENCHMARK_CONFIGS):
    """Returns (summary_df, per_query_df)."""
    max_k = max(ks)
    rows, per_query = [], []
    for label, method, strategy in configs:
        chunks = engine.chunk_sets[strategy]
        latencies, metrics = [], {f"{m}@{k}": [] for k in ks for m in ("Hit", "Recall", "MRR")}
        for q in queries:
            rel = relevant_ids(chunks, engine.docs[q.doc], q)
            resp = engine.search(q.query, method, strategy, k=max_k)
            ranked = [r.chunk_id for r in resp.results]
            latencies.append(resp.latency_ms)
            for k in ks:
                metrics[f"Hit@{k}"].append(hit_at_k(ranked, rel, k))
                metrics[f"Recall@{k}"].append(recall_at_k(ranked, rel, k))
                metrics[f"MRR@{k}"].append(reciprocal_rank_at_k(ranked, rel, k))
            first = next((i + 1 for i, r in enumerate(ranked) if r in rel), None)
            per_query.append({"config": label, "method": method, "query_id": q.id, "type": q.type, "query": q.query, "first_relevant_rank": first})
        row = {"config": label, "method": method, "chunking": strategy, "chunks": len(chunks),
               "avg_tokens": round(float(np.mean([c.token_count for c in chunks])), 1)}
        row.update({name: round(float(np.mean(v)), 3) for name, v in metrics.items()})
        row.update({"p50 ms": round(percentile(latencies, 50), 2), "p95 ms": round(percentile(latencies, 95), 2)})
        for t in ("lexical", "semantic"):
            idx = [i for i, q in enumerate(queries) if q.type == t]
            row[f"Hit@5 {t}"] = round(float(np.mean([metrics["Hit@5"][i] for i in idx])), 3) if idx and 5 in ks else None
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(per_query)


def failure_analysis(engine: SearchEngine, queries: Sequence[EvalQuery], per_query: pd.DataFrame, k: int = 5,
                     lexical_cfg: str = BENCHMARK_CONFIGS[0][0], dense_cfg: str = "Dense · structure-aware") -> pd.DataFrame:
    """Queries where one retriever found the answer in the top-k and the other did not, with a reason."""
    by_q = {q.id: q for q in queries}
    pq = per_query.set_index(["config", "query_id"])["first_relevant_rank"]
    out = []
    for qid, q in by_q.items():
        b, d = pq.get((lexical_cfg, qid)), pq.get((dense_cfg, qid))
        b_hit = b is not None and not pd.isna(b) and b <= k
        d_hit = d is not None and not pd.isna(d) and d <= k
        if b_hit == d_hit:
            continue
        s0, s1 = locate_span(engine.docs[q.doc], q.answer_span)
        overlap = keyword_overlap(q.query, engine.docs[q.doc][max(0, s0 - 400) : s1 + 400])
        rare = [t for t in tokenize(q.query) if "_" in t or (any(ch.isdigit() for ch in t)) or len(t) > 12]
        if b_hit:
            why = (f"Exact identifier(s) {rare} are rare tokens with high IDF, so BM25 ranks them first; "
                   "the embedding blurs them with similar-looking names.") if rare else (
                   f"Query shares precise terms {overlap[:6]} with the answer; BM25 rewards exact term matches.")
        else:
            why = (f"Paraphrase: only {len(overlap)} shared keyword(s) {overlap[:4]}; BM25 has little to match, "
                   "while the embedding matches the meaning.")
        out.append({"query_id": qid, "type": q.type, "query": q.query, "winner": "BM25" if b_hit else "Dense",
                    "BM25 rank": None if pd.isna(b) else int(b), "Dense rank": None if pd.isna(d) else int(d), "why": why})
    return pd.DataFrame(out)
