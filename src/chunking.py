"""Chunking strategies for Markdown technical documentation.

Two strategies are compared in the benchmark:

* ``FixedSizeChunker``: slides a window of N tokens (tiktoken ``cl100k_base``)
  with a configurable overlap. Simple and predictable, but blind to structure.
* ``StructureAwareChunker``: splits on Markdown headers (``#`` to ``####``),
  never inside a code fence, and keeps the header breadcrumb with every chunk.

Every chunk records the character span it came from (``char_start`` /
``char_end``) in its source file. Evaluation uses those spans, not chunk list
positions, so ground truth stays valid no matter how the text is chunked.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import tiktoken
from pydantic import BaseModel, Field

ENCODING_NAME = "cl100k_base"  # tokenizer used by text-embedding-3-small
HEADER_RE = re.compile(r"^(#{1,4})\s+(.*\S)\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)(\S*)")


@lru_cache(maxsize=1)
def get_encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str) -> int:
    return len(get_encoding().encode(text, disallowed_special=()))


class DocumentChunk(BaseModel):
    """One retrievable unit of text."""

    chunk_id: str
    content: str
    parent_doc: str
    token_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def header_path(self) -> str:
        return " > ".join(self.metadata.get("header_path", []))


# --------------------------------------------------------------------------- #
# Markdown structure helpers
# --------------------------------------------------------------------------- #
def clean_header(text: str) -> str:
    return text.replace("`", "").strip()


class LineInfo(BaseModel):
    """Per-line structure facts, computed once per document."""

    starts: list[int]  # character offset where each line starts
    in_fence: list[bool]  # True if the line is inside (or opens/closes) a code fence
    fence_lang: list[str]  # language of the fence this line belongs to
    header_path: list[list[str]]  # header breadcrumb in effect at each line
    is_header: list[bool]


def analyze_lines(text: str) -> LineInfo:
    lines = text.split("\n")
    starts, in_fence, langs, paths, is_header = [], [], [], [], []
    pos, fence_open, lang, stack = 0, False, "", []  # stack of (level, title)
    for line in lines:
        starts.append(pos)
        pos += len(line) + 1
        m_fence = FENCE_RE.match(line)
        if m_fence:
            if not fence_open:
                fence_open, lang = True, m_fence.group(2)
                in_fence.append(True)
            else:
                in_fence.append(True)
                fence_open = False
            langs.append(lang)
            paths.append([t for _, t in stack])
            is_header.append(False)
            continue
        m_head = None if fence_open else HEADER_RE.match(line)
        if m_head:
            level = len(m_head.group(1))
            stack = [(lv, t) for lv, t in stack if lv < level] + [(level, clean_header(m_head.group(2)))]
        in_fence.append(fence_open)
        langs.append(lang if fence_open else "")
        paths.append([t for _, t in stack])
        is_header.append(bool(m_head))
    return LineInfo(starts=starts, in_fence=in_fence, fence_lang=langs, header_path=paths, is_header=is_header)


def line_of(info: LineInfo, char_offset: int) -> int:
    """0-based line index containing ``char_offset``."""
    return max(0, bisect_right(info.starts, char_offset) - 1)


def fence_state_at(text: str, info: LineInfo, char_offset: int) -> tuple[bool, str]:
    """Is ``char_offset`` inside an open code fence? Returns (open, language).

    Counts fence lines that start before the offset; an odd count means a fence
    is open. The language is taken from the last opening fence.
    """
    lines = text.split("\n")
    open_, lang = False, ""
    for idx, start in enumerate(info.starts):
        if start >= char_offset:
            break
        m = FENCE_RE.match(lines[idx])
        if m:
            open_, lang = (True, m.group(2)) if not open_ else (False, "")
    return open_, lang


def repair_fences(content: str, starts_in_fence: bool, lang: str) -> tuple[str, bool]:
    """Close/reopen code fences cut by a fixed window so every chunk is valid Markdown."""
    repaired = False
    if starts_in_fence:
        content = f"```{lang}\n{content}"
        repaired = True
    fences = sum(1 for ln in content.split("\n") if FENCE_RE.match(ln))
    if fences % 2 == 1:
        content = f"{content}\n```"
        repaired = True
    return content, repaired


# --------------------------------------------------------------------------- #
# Chunkers
# --------------------------------------------------------------------------- #
class FixedSizeChunker:
    """Fixed token windows with overlap: chunk i covers tokens [i*step, i*step+size)."""

    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 32, repair_code_fences: bool = True):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.repair_code_fences = repair_code_fences
        self.name = f"fixed-{chunk_size}"

    def split(self, text: str, doc_name: str) -> list[DocumentChunk]:
        enc = get_encoding()
        tokens = enc.encode(text, disallowed_special=())
        # character offset of every token boundary (exact, via token bytes)
        byte_ends, total = [], 0
        for t in tokens:
            total += len(enc.decode_single_token_bytes(t))
            byte_ends.append(total)
        raw = text.encode("utf-8")
        info = analyze_lines(text)

        def char_at(byte_offset: int) -> int:
            return len(raw[:byte_offset].decode("utf-8", errors="ignore"))

        step = self.chunk_size - self.chunk_overlap
        chunks: list[DocumentChunk] = []
        for i, start in enumerate(range(0, max(len(tokens), 1), step)):
            end = min(start + self.chunk_size, len(tokens))
            b0 = byte_ends[start - 1] if start else 0
            b1 = byte_ends[end - 1] if end else 0
            c0, c1 = char_at(b0), char_at(b1)
            body = text[c0:c1]
            first_line, last_line = line_of(info, c0), line_of(info, max(c0, c1 - 1))
            starts_in_fence, lang = fence_state_at(text, info, c0)
            content, repaired = (body, False)
            if self.repair_code_fences:
                content, repaired = repair_fences(body, starts_in_fence, lang)
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_name}::{self.name}::{i:04d}",
                    content=content,
                    parent_doc=doc_name,
                    token_count=end - start,
                    metadata={
                        "strategy": self.name,
                        "token_start": start,
                        "token_end": end,
                        "char_start": c0,
                        "char_end": c1,
                        "line_start": first_line + 1,
                        "line_end": last_line + 1,
                        "header_path": info.header_path[first_line],
                        "fence_repaired": repaired,
                    },
                )
            )
            if end >= len(tokens):
                break
        return chunks


class StructureAwareChunker:
    """Split on Markdown headers; never inside a code fence; keep header breadcrumbs.

    Sections larger than ``max_tokens`` are split further at blank lines (then at
    single lines), but a code fence is never cut: an oversized code block stays
    whole in its own chunk.
    """

    def __init__(self, max_tokens: int = 512):
        self.max_tokens = max_tokens
        self.name = "structure"

    def _sections(self, text: str, info: LineInfo) -> list[tuple[int, int]]:
        """(first_line, last_line_exclusive) ranges, each starting at a header."""
        bounds = [i for i, h in enumerate(info.is_header) if h]
        if not bounds or bounds[0] != 0:
            bounds = [0] + bounds
        n = len(info.starts)
        return [(b, bounds[k + 1] if k + 1 < len(bounds) else n) for k, b in enumerate(bounds)]

    def _blocks(self, text: str, info: LineInfo, lo: int, hi: int) -> list[tuple[int, int]]:
        """Break a line range into atomic blocks: paragraphs and whole code fences."""
        lines = text.split("\n")
        blocks, cur = [], lo
        i = lo
        while i < hi:
            if FENCE_RE.match(lines[i]):  # swallow the whole fenced block
                j = i + 1
                while j < hi and not FENCE_RE.match(lines[j]):
                    j += 1
                i = min(j + 1, hi)
                continue
            if lines[i].strip() == "" and i > cur:
                blocks.append((cur, i + 1))
                cur = i + 1
            i += 1
        if cur < hi:
            blocks.append((cur, hi))
        # an oversized non-code block (e.g. a long table) may be cut at line level
        out = []
        for a, b in blocks:
            block_text = "\n".join(lines[a:b])
            if count_tokens(block_text) > self.max_tokens and not any(FENCE_RE.match(l) for l in lines[a:b]):
                out.extend((k, k + 1) for k in range(a, b))
            else:
                out.append((a, b))
        return out

    def split(self, text: str, doc_name: str) -> list[DocumentChunk]:
        info = analyze_lines(text)
        lines = text.split("\n")
        chunks: list[DocumentChunk] = []
        carry: tuple[int, int] | None = None  # tiny header-only section waiting to merge

        def emit(a: int, b: int) -> None:
            body = "\n".join(lines[a:b]).strip("\n")
            if not body.strip():
                return
            # use the deepest header of the leading header block (merged bare headers)
            deepest = a
            for i in range(a, b):
                t = lines[i].strip()
                if info.is_header[i]:
                    deepest = i
                elif t and not re.fullmatch(r"<a [^>]*></a>", t):
                    break
            path = info.header_path[deepest] or [doc_name.removesuffix(".md")]
            crumb = " > ".join(path)
            content = f"[{doc_name} | {crumb}]\n{body}"
            c0 = info.starts[a]
            c1 = info.starts[b - 1] + len(lines[b - 1])
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_name}::{self.name}::{len(chunks):04d}",
                    content=content,
                    parent_doc=doc_name,
                    token_count=count_tokens(content),
                    metadata={
                        "strategy": self.name,
                        "char_start": c0,
                        "char_end": c1,
                        "line_start": a + 1,
                        "line_end": b,
                        "header_path": path,
                        "fence_repaired": False,
                    },
                )
            )

        for a, b in self._sections(text, info):
            if carry:
                a = carry[0]
                carry = None
            sec_text = "\n".join(lines[a:b])
            if _is_header_only(lines[a:b]) and b < len(lines):
                carry = (a, b)  # a bare header (no body) is merged into the next section
                continue
            if count_tokens(sec_text) <= self.max_tokens:
                emit(a, b)
                continue
            # pack atomic blocks greedily up to max_tokens
            pack_start, pack_tokens = a, 0
            for x, y in self._blocks(text, info, a, b):
                t = count_tokens("\n".join(lines[x:y]))
                if pack_tokens and pack_tokens + t > self.max_tokens:
                    emit(pack_start, x)
                    pack_start, pack_tokens = x, 0
                pack_tokens += t
            emit(pack_start, b)
        if carry:
            emit(*carry)
        return chunks


def _is_header_only(section_lines: list[str]) -> bool:
    """True if a section has no body: only headers, HTML anchors and blank lines."""
    for ln in section_lines:
        t = ln.strip()
        if t and not HEADER_RE.match(ln) and not re.fullmatch(r"<a [^>]*></a>", t):
            return False
    return True


# --------------------------------------------------------------------------- #
# Corpus loading
# --------------------------------------------------------------------------- #
def load_corpus(corpus_dir: str | Path) -> dict[str, str]:
    """{file_name: text} for every Markdown file in the corpus folder."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(corpus_dir).rglob("*.md"))}


def chunk_corpus(docs: dict[str, str], chunker: FixedSizeChunker | StructureAwareChunker) -> list[DocumentChunk]:
    out: list[DocumentChunk] = []
    for name, text in docs.items():
        out.extend(chunker.split(text, name))
    return out


def fences_balanced(content: str) -> bool:
    return sum(1 for ln in content.split("\n") if FENCE_RE.match(ln)) % 2 == 0


def iter_fence_blocks(text: str) -> Iterable[str]:
    """Yield every complete fenced code block (used by tests)."""
    lines, buf, open_ = text.split("\n"), [], False
    for ln in lines:
        if FENCE_RE.match(ln):
            buf.append(ln)
            if open_:
                yield "\n".join(buf)
                buf = []
            open_ = not open_
        elif open_:
            buf.append(ln)
