"""SearchWorkbench: live BM25-vs-dense comparator + benchmark dashboard.

Run locally:  uv run streamlit run src/app.py
"""

from __future__ import annotations

import html
import os
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.benchmarker import failure_analysis, load_eval, run_benchmark  # noqa: E402
from src.demo_queries import DEMO_QUERIES  # noqa: E402
from src.embeddings import CachedEmbedder, MissingEmbeddingError, OpenAIEmbedder  # noqa: E402
from src.search import CHUNKERS, SearchEngine  # noqa: E402

try:  # Streamlit Cloud keeps the key in its secret manager
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = str(st.secrets["OPENAI_API_KEY"]).strip()
except Exception:  # no secrets.toml locally: .env is used instead
    pass

st.set_page_config(page_title="SearchWorkbench", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Chunking corpus and building indexes…")
def get_engine() -> SearchEngine:
    has_key = bool(os.getenv("OPENAI_API_KEY", "").strip())
    embedder = CachedEmbedder(fallback=OpenAIEmbedder() if has_key else None)
    engine = SearchEngine.from_corpus(embedder=embedder)
    for name in CHUNKERS:  # build every index up front so timings exclude indexing
        engine.bm25(name)
        engine.dense(name)
    return engine


def highlight(text: str, terms: list[str]) -> str:
    safe = html.escape(text[:900] + ("…" if len(text) > 900 else ""))
    for t in sorted(terms, key=len, reverse=True):
        if len(t) > 2:
            safe = re.sub(rf"(?i)\b({re.escape(t)})\b", r"<mark>\1</mark>", safe)
    return safe.replace("\n", "<br>")


def show_results(title: str, resp, score_label: str) -> None:
    st.markdown(f"#### {title}")
    st.caption(f"{len(resp.results)} results · {resp.latency_ms:.1f} ms")
    for r in resp.results:
        with st.container(border=True):
            st.markdown(f"**#{r.rank}** · `{r.parent_doc}` › {r.header_path or '—'}")
            st.markdown(f"{score_label}: **{r.score:.3f}** · matched terms: {', '.join(r.matched_terms) or '— none —'}")
            st.markdown(f"<div style='font-size:0.85rem'>{highlight(r.content, r.matched_terms)}</div>", unsafe_allow_html=True)


try:
    engine = get_engine()
except MissingEmbeddingError as exc:
    st.error(f"Embeddings missing: {exc}")
    st.stop()

st.title("🔎 SearchWorkbench")
st.caption(
    f"Fastify reference docs · {len(engine.docs)} files · "
    f"{sum(len(t) for t in engine.docs.values()):,} characters · text-embedding-3-small + ChromaDB vs. BM25"
)

tab_search, tab_bench = st.tabs(["Live Search Comparator", "Benchmark & Evaluation"])

# --------------------------------------------------------------------------- #
with tab_search:
    st.markdown("**Try an example:**")
    cols = st.columns(3)
    for i, q in enumerate(DEMO_QUERIES):
        if cols[i % 3].button(q, key=f"demo{i}", width="stretch"):
            st.session_state["query"] = q
    query = st.text_input("Search query", key="query", placeholder="e.g. FST_ERR_HOOK_TIMEOUT or 'how do I hide secrets in logs?'")
    c1, c2 = st.columns(2)
    k = c1.slider("Top K", 1, 10, 5)
    strategy = c2.selectbox("Chunking strategy (both engines)", list(CHUNKERS), index=list(CHUNKERS).index("structure"))
    if query:
        left, right = st.columns(2)
        with left:
            show_results("🔤 BM25 keyword search", engine.search(query, "bm25", strategy, k), "BM25 score")
        with right:
            try:
                show_results("🧠 Dense vector search", engine.search(query, "dense", strategy, k), "Cosine similarity")
            except MissingEmbeddingError:
                st.warning("This query isn't in the embedding cache and no OPENAI_API_KEY is set. Try an example query.")

# --------------------------------------------------------------------------- #
with tab_bench:
    queries = load_eval()
    st.markdown(
        f"**{len(queries)} labelled queries** ({sum(q.type == 'lexical' for q in queries)} exact-keyword, "
        f"{sum(q.type == 'semantic' for q in queries)} conceptual). A result counts as relevant if its source "
        "text overlaps the labelled answer span, so labels stay valid for every chunk size."
    )
    if st.button("▶️ Run Benchmark", type="primary"):
        with st.spinner("Running every query against every configuration…"):
            summary, per_query = run_benchmark(engine, queries)
            st.session_state["bench"] = (summary, per_query, failure_analysis(engine, queries, per_query))
    if "bench" in st.session_state:
        summary, per_query, failures = st.session_state["bench"]
        st.subheader("Results")
        show = ["config", "chunks", "avg_tokens", "Hit@3", "Hit@5", "Hit@10", "Recall@5", "MRR@5", "MRR@10",
                "Hit@5 lexical", "Hit@5 semantic", "p50 ms", "p95 ms"]
        st.dataframe(summary[show], hide_index=True, width="stretch")
        best = summary.sort_values(["MRR@5", "Hit@5"], ascending=False).iloc[0]
        st.success(f"Best MRR@5: **{best['config']}** (MRR@5 = {best['MRR@5']:.2f}, Hit@5 = {best['Hit@5']:.2f})")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Hit Rate@K and MRR@5 by configuration**")
            st.bar_chart(summary.set_index("config")[["Hit@3", "Hit@5", "Hit@10", "MRR@5"]], stack=False)
        with c2:
            st.markdown("**Hit@5 by query type**")
            st.bar_chart(summary.set_index("config")[["Hit@5 lexical", "Hit@5 semantic"]], stack=False)
            st.markdown("**Search latency (ms, query embedding cached)**")
            st.bar_chart(summary.set_index("config")[["p50 ms", "p95 ms"]], stack=False)

        st.subheader("Failure analysis: BM25 vs. Dense (structure-aware), top-5")
        if failures.empty:
            st.info("Both engines agree on every query.")
        else:
            st.dataframe(failures, hide_index=True, width="stretch")

        with st.expander("Per-query ranks (first relevant result; empty = not in top 10)"):
            st.dataframe(per_query.pivot(index=["query_id", "type", "query"], columns="config", values="first_relevant_rank"), width="stretch")
