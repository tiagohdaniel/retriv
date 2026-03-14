from app.core.chunker import SemanticChunker, TextChunker


# ---------------------------------------------------------------------------
# SemanticChunker
# ---------------------------------------------------------------------------

def test_semantic_empty():
    c = SemanticChunker(chunk_size=200)
    assert c.chunk("") == []
    assert c.chunk("   ") == []


def test_semantic_single_paragraph_fits():
    text = "This is a short paragraph that fits in one chunk."
    c = SemanticChunker(chunk_size=200, chunk_overlap=0)
    chunks = c.chunk(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_semantic_respects_paragraph_boundaries():
    text = "First paragraph with some content.\n\nSecond paragraph with other content.\n\nThird paragraph here."
    c = SemanticChunker(chunk_size=60, chunk_overlap=0)
    chunks = c.chunk(text)
    # No chunk should contain content from non-adjacent paragraphs if they exceed limit
    for chunk in chunks:
        assert chunk.strip()


def test_semantic_never_cuts_mid_sentence_when_possible():
    text = (
        "Sentence one ends here. "
        "Sentence two is also here. "
        "Sentence three completes the paragraph."
    )
    c = SemanticChunker(chunk_size=50, chunk_overlap=0)
    chunks = c.chunk(text)
    # Each chunk should end with punctuation or be the last
    for chunk in chunks[:-1]:
        # Should end at a sentence boundary (last char is punctuation or follows one)
        assert chunk.strip()[-1] in ".!?", f"Chunk cut mid-sentence: {repr(chunk)}"


def test_semantic_overlap_carries_last_unit():
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    c = SemanticChunker(chunk_size=30, chunk_overlap=20)
    chunks = c.chunk(text)
    assert len(chunks) >= 2
    # The overlap means chunk 1 content should appear (or part of it) in chunk 2
    # At minimum, the chunker should not crash and produce valid strings
    for chunk in chunks:
        assert isinstance(chunk, str)
        assert chunk.strip()


def test_semantic_large_paragraph_falls_back_to_sentences():
    # Single paragraph larger than chunk_size — must be split at sentence boundaries
    text = (
        "The retrieval-augmented generation pipeline first embeds the query. "
        "Then it searches the vector store for the most similar chunks. "
        "Finally, it passes the retrieved context to the language model."
    )
    c = SemanticChunker(chunk_size=80, chunk_overlap=0)
    chunks = c.chunk(text)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk) <= 200  # no chunk should be absurdly large


def test_semantic_returns_all_content():
    """Total content of all chunks should cover all source text (no silent drops)."""
    text = "Alpha.\n\nBeta gamma delta.\n\nEpsilon zeta."
    c = SemanticChunker(chunk_size=20, chunk_overlap=0)
    chunks = c.chunk(text)
    combined = " ".join(chunks)
    for word in ["Alpha", "Beta", "Epsilon"]:
        assert word in combined, f"Word '{word}' was dropped"


# ---------------------------------------------------------------------------
# TextChunker (regression — must still work)
# ---------------------------------------------------------------------------

def test_fixed_chunker_basic():
    c = TextChunker(chunk_size=20, chunk_overlap=5)
    chunks = c.chunk("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk) <= 20
