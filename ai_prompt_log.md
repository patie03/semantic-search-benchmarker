# AI Prompt Log & Code Audit

**Student:** Patience Kalanga
**AI assistant used:** Claude (Anthropic), in a chat-based coding session
**Date:** November 2026

This log records the prompt, what the AI produced, the bugs found when the draft was audited and tested, and how they were fixed.

---

## 1. Prompt used

I started from the course's sample prompt and adapted it:

```text
You are a Staff Information Retrieval and Search Engineer. I need a modular Python 3.11+ semantic
search and evaluation pipeline that indexes technical Markdown documentation, evaluates multiple
chunking strategies, and benchmarks dense vector retrieval against BM25 keyword search.

Ingestion & Chunking (src/chunking.py):
 - FixedSizeChunker(chunk_size, chunk_overlap) that splits text into tokens with tiktoken.
 - StructureAwareChunker() that splits on Markdown headers (#, ##, ###) while keeping code
   blocks (```) intact inside their parent sections.
 - Each chunk is a Pydantic model DocumentChunk: chunk_id, content, metadata (header hierarchy,
   line numbers), parent_doc.

Dual Indexing (src/indexer.py):
 - Dense: ChromaDB with text-embedding-3-small; normalize embeddings to unit length for cosine.
 - Sparse: rank_bm25.BM25Okapi with lower-cased, punctuation-stripped tokens.

Benchmark Engine (src/benchmarker.py):
 - Load eval_queries.json; compute Hit Rate@K, Recall@K, MRR@K, p50/p95 latency.
 - Compare BM25, Dense 256-token chunks, Dense 1024-token chunks, Dense structure-aware.

My additions:
 - Corpus: the Fastify reference docs (MIT licensed).
 - Tests must run without an API key, so cache the embeddings on disk.
 - Also build a Streamlit dashboard with a side-by-side search and a benchmark tab.
```

## 2. What the AI produced

Modules `chunking.py`, `embeddings.py`, `indexer.py`, `search.py`, `benchmarker.py` and `app.py`, an evaluation set of 20 labelled queries, and a pytest suite.

## 3. Bugs found in the AI draft (with before/after)

### Bug 1: The structure-aware chunker lost header breadcrumbs and merged unrelated sections
Found by `test_structure_chunker_keeps_header_breadcrumbs`.

- **What happened:** the draft merged any section under 24 tokens into the next one. Short but real sections (like a one-line flag description) were glued to their neighbours. The merged chunk kept the breadcrumb of its *first* line, so `### CUSTOM_CONFIG_FLAG_XYZ` was labelled only `Config Reference`. On the small fixture, BM25 then ranked the wrong chunk first for the flag name.

**Before**
```python
if count_tokens(sec_text) < self.min_tokens and b < len(lines):
    carry = (a, b)  # merge a short section into the next section
...
path = info.header_path[a]
```
**After**
```python
if _is_header_only(lines[a:b]) and b < len(lines):
    carry = (a, b)  # only a bare header (no body) is merged forward
...
# use the deepest header of the leading header block
path = info.header_path[deepest]
```

### Bug 2: The fixed-size chunker cut code fences in half
Found by `test_fixed_chunker_repairs_cut_fences`.

- **What happened:** a token window ends wherever the token count says, often in the middle of a ```` ```js ```` block. That chunk has an unclosed fence, and the next chunk starts with code but no opening fence, so the code loses its meaning. With 256-token windows, **178 of 361 chunks** were cut this way.

**Before**
```python
content = text[c0:c1]  # raw window
```
**After**
```python
starts_in_fence, lang = fence_state_at(text, info, c0)
content, repaired = repair_fences(body, starts_in_fence, lang)  # re-open ```js / close ```
```
The raw token range is still stored in metadata, so the overlap test checks exact token overlap independently of the repair.

### Bug 3: The fence-state helper had a dead branch, so it was always wrong for one case
Found in code review.

**Before**
```python
return prev_open or (started_mid_line and not prev_open and False)
```
`... and False` is always `False`, so a window starting in the middle of a fence line was never detected. The helper was also tangled across three functions.

**After:** one simple function, `fence_state_at()`, counts fence lines that start before the offset. An odd count means a fence is open, and the language comes from the last opener.

## 4. Audit checklist from the assignment

| Risk | Result in this project |
|---|---|
| **Ground-truth leakage via chunk index** | ✅ Avoided. Labels are an exact *answer span* in a source file. For every chunking strategy, a chunk is relevant if its source character range overlaps the span (`relevant_ids()`), so changing the chunk size can't break the labels. |
| **Cosine distance vs. similarity inversion** | ✅ Checked. Chroma returns cosine **distance**, which the code converts with `similarity = 1 − distance`. `test_dense_scores_are_similarities_highest_first` asserts that identical text gets ≈1.0 and ranks first. |
| **BM25 tokenization asymmetry** | ✅ Checked. One `tokenize()` function is used for documents **and** queries. It lower-cases, strips punctuation, keeps `fst_err_hook_timeout` whole and adds camelCase parts. Covered by `test_bm25_query_and_document_tokenization_are_symmetric`. |
| **Code blocks sliced by the naive chunker** | ❌ → ✅ Bug 2 above. The structure-aware chunker never cuts a fence (an oversized code block stays whole, which is why the max chunk is 767 tokens). |
| **MRR math (outside top-K, division by zero)** | ✅ Checked. Only the top-K count, a miss scores `0.0`, and duplicate IDs are counted once. Covered by the 7-case table in `test_metric_math`. |

## 5. Known limitations (honest notes)
- **Bigger chunks are more likely to "contain" the answer.** That raises Hit Rate for 1024-token chunks but gives the reader much more text to go through. MRR and precision should be read together.
- **Latency excludes the embedding API call**, because query vectors are cached. In live use, add about 100–300 ms for the OpenAI request.
- **The evaluation set has only 20 queries**, so a single query changes Hit@5 by 0.05.

## 6. My verification (tick each box after doing it myself)
- [ ] Ran `uv run python scripts/build_embeddings.py` to create the embedding cache
- [ ] Ran `uv run pytest -v`: all tests pass
- [ ] Ran the app, compared an exact query and a paraphrased query side by side
- [ ] Ran the benchmark and checked the numbers in the README
