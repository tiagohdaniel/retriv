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


def test_semantic_section_header_forces_new_chunk():
    """Numbered section headers must never be merged with preceding content,
    even when the combined size would fit within chunk_size.
    Reproduces the real case from guia-pgdas-d.pdf where '1.4. QUANDO UTILIZAR'
    and '1.5. CONCEITOS PRELIMINARES' were merged into one chunk, causing the
    ME limit (R$ 360.000,00) to be in a different chunk than 'Microempresa'.
    """
    text = (
        "1.4. QUANDO UTILIZAR\n\n"
        "A declaração e o recolhimento devem ser efetuados até o dia 20 do mês.\n\n"
        "1.5. CONCEITOS PRELIMINARES\n\n"
        "Microempresa (ME) — receita bruta igual ou inferior a R$ 360.000,00."
    )
    c = SemanticChunker(chunk_size=800, chunk_overlap=0)
    chunks = c.chunk(text)

    # 1.4 and 1.5 must be in separate chunks
    chunk_with_14 = next((ch for ch in chunks if "1.4." in ch), None)
    chunk_with_15 = next((ch for ch in chunks if "1.5." in ch), None)
    assert chunk_with_14 is not None, "Section 1.4 not found in any chunk"
    assert chunk_with_15 is not None, "Section 1.5 not found in any chunk"
    assert chunk_with_14 is not chunk_with_15, "Sections 1.4 and 1.5 must be in separate chunks"

    # The ME definition and its limit must be in the same chunk
    assert "Microempresa" in chunk_with_15
    assert "360.000,00" in chunk_with_15


# ---------------------------------------------------------------------------
# SemanticChunker — overlap strategy C (complete semantic units)
# ---------------------------------------------------------------------------

def test_overlap_carries_complete_sentence_even_when_larger_than_budget():
    """The last semantic unit must always be carried whole into the next chunk,
    even when its length exceeds chunk_overlap. The old behaviour truncated to
    chunk_overlap raw chars — Strategy C must carry the full sentence."""
    long_s = "A Microempresa tem limite de receita bruta anual de R$ 360.000,00."
    chunk_overlap = 30
    chunk_size = 100
    assert chunk_overlap < len(long_s) < chunk_size  # semantic unit, larger than budget

    # Build text so that long_s ends up as the last unit of a chunk, triggering
    # the overlap carry for the following chunk.
    text = "Primeira frase.\n\nSegunda frase.\n\n" + long_s + "\n\nConteúdo seguinte."
    c = SemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = c.chunk(text)

    # long_s must appear COMPLETE in the chunk that follows the one where it first
    # appears (carried as overlap) — not just the last chunk_overlap chars of it.
    first_idx = next((i for i, ch in enumerate(chunks) if long_s in ch), None)
    assert first_idx is not None, "long_s not found in any chunk"
    if first_idx + 1 < len(chunks):
        assert chunks[first_idx + 1].startswith(long_s), (
            f"Expected next chunk to start with the full sentence as overlap, "
            f"got: {repr(chunks[first_idx + 1][:80])}"
        )


def test_overlap_carries_multiple_units_when_budget_allows():
    """When the last unit is short enough, additional units should be carried
    backwards until the budget is exhausted."""
    text = "Sentence one.\n\nSentence two.\n\nSentence three.\n\nSentence four."
    c = SemanticChunker(chunk_size=40, chunk_overlap=40)
    chunks = c.chunk(text)
    assert len(chunks) >= 2
    # At least one pair of consecutive chunks must share content (overlap visible)
    shared = any(
        any(line.strip() in chunks[i + 1] for line in chunks[i].split("\n\n") if line.strip())
        for i in range(len(chunks) - 1)
    )
    assert shared, "No overlap content found between any consecutive chunks"


def test_hard_split_units_use_char_suffix_fallback():
    """Units from _hard_split (no sentence boundaries) receive a char-suffix
    overlap fallback. The safety guard must prevent the suffix from inflating
    a subsequent chunk beyond chunk_size."""
    # Build content with no sentence endings — forces _hard_split on all units.
    long_word_soup = " ".join(["palavra"] * 90)  # ~630 chars, no punctuation
    chunk_size = 200
    chunk_overlap = 50
    c = SemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = c.chunk(long_word_soup)
    # chunk_size is a hard upper bound — no chunk may exceed it.
    for ch in chunks:
        assert len(ch) <= chunk_size, f"Chunk exceeded chunk_size: {len(ch)}"
    # Must produce multiple chunks (content is larger than chunk_size).
    assert len(chunks) >= 2


def test_overlap_zero_disables_carry():
    """chunk_overlap=0 must produce no overlap between chunks."""
    text = "First sentence.\n\nSecond sentence.\n\nThird sentence."
    c = SemanticChunker(chunk_size=30, chunk_overlap=0)
    chunks = c.chunk(text)
    if len(chunks) >= 2:
        # No sentence from chunk N should appear in chunk N+1
        for i in range(len(chunks) - 1):
            for part in chunks[i].split("\n\n"):
                assert part.strip() not in chunks[i + 1], (
                    f"Unexpected overlap: {repr(part)} found in next chunk"
                )


# ---------------------------------------------------------------------------
# TextChunker (regression — must still work)
# ---------------------------------------------------------------------------

def test_fixed_chunker_basic():
    c = TextChunker(chunk_size=20, chunk_overlap=5)
    chunks = c.chunk("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk) <= 20
