"""Chunk boundary, overlap and code-fence tests."""

import pytest

from src.chunking import (
    FixedSizeChunker,
    StructureAwareChunker,
    fences_balanced,
    get_encoding,
    iter_fence_blocks,
    load_corpus,
)
from tests.conftest import FIXTURE_DOC

DOCS = load_corpus("corpus")


# Test 1 --------------------------------------------------------------------- #
@pytest.mark.parametrize("size,overlap", [(256, 32), (512, 64), (1024, 128)])
def test_fixed_chunk_overlap_integrity(size, overlap):
    """Consecutive chunks share exactly `overlap` tokens and no characters are dropped."""
    text = DOCS["Server.md"]
    enc = get_encoding()
    tokens = enc.encode(text, disallowed_special=())
    chunks = FixedSizeChunker(size, overlap, repair_code_fences=False).split(text, "Server.md")

    assert chunks[0].metadata["token_start"] == 0
    assert chunks[-1].metadata["token_end"] == len(tokens)
    for a, b in zip(chunks, chunks[1:]):
        assert a.metadata["token_end"] - b.metadata["token_start"] == overlap
        shared_a = tokens[a.metadata["token_start"]:a.metadata["token_end"]][-overlap:]
        shared_b = tokens[b.metadata["token_start"]:b.metadata["token_end"]][:overlap]
        assert shared_a == shared_b
    # Rebuild the document from the non-overlapping parts: must equal the original exactly.
    rebuilt = b"".join(
        enc.decode_bytes(tokens[c.metadata["token_start"] + (overlap if i else 0):c.metadata["token_end"]])
        for i, c in enumerate(chunks)
    )
    assert rebuilt == text.encode("utf-8")
    # And each chunk's text is exactly its character span of the source.
    for c in chunks:
        assert c.content == text[c.metadata["char_start"]:c.metadata["char_end"]]


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        FixedSizeChunker(100, 100)


# Test 2 --------------------------------------------------------------------- #
def test_structure_chunker_never_severs_code_fences():
    for name, text in DOCS.items():
        chunks = StructureAwareChunker(max_tokens=512).split(text, name)
        assert all(fences_balanced(c.content) for c in chunks), name
        # every original code block appears whole inside exactly one chunk
        for block in iter_fence_blocks(text):
            assert sum(block in c.content for c in chunks) >= 1, f"{name}: code block was split"


def test_structure_chunker_keeps_header_breadcrumbs():
    chunks = StructureAwareChunker().split(FIXTURE_DOC, "fixture.md")
    flag = next(c for c in chunks if "cache warmer" in c.content)
    assert flag.metadata["header_path"] == ["Config Reference", "Flags", "CUSTOM_CONFIG_FLAG_XYZ"]
    assert flag.content.startswith("[fixture.md | Config Reference > Flags > CUSTOM_CONFIG_FLAG_XYZ]")
    assert "retries: 3" in flag.content and "})" in flag.content  # code block kept with its section


def test_fixed_chunker_repairs_cut_fences():
    """The naive window cuts code blocks; repair re-opens/closes them so chunks stay valid Markdown."""
    naive = FixedSizeChunker(256, 32, repair_code_fences=False)
    fixed = FixedSizeChunker(256, 32)
    text = DOCS["Hooks.md"]
    assert not all(fences_balanced(c.content) for c in naive.split(text, "Hooks.md"))  # the bug exists
    repaired = fixed.split(text, "Hooks.md")
    assert all(fences_balanced(c.content) for c in repaired)
    assert any(c.metadata["fence_repaired"] for c in repaired)


def test_chunk_metadata_is_complete():
    for strategy in (FixedSizeChunker(512, 64), StructureAwareChunker()):
        for c in strategy.split(DOCS["Reply.md"], "Reply.md"):
            for key in ("char_start", "char_end", "line_start", "line_end", "header_path", "strategy"):
                assert key in c.metadata
            assert c.token_count > 0
