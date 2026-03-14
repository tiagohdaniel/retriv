import re


class TextChunker:
    """Fixed-size character chunker (legacy — use SemanticChunker for better quality).

    Splits text into overlapping fixed-size chunks. Simple and predictable,
    but may cut sentences in the middle, losing context at boundaries.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = chunk_overlap

    def chunk(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = end - self.overlap

        return chunks


class SemanticChunker:
    """Paragraph- and sentence-aware chunker.

    Strategy:
    1. Split text at paragraph boundaries (double newlines, markdown headers).
    2. Paragraphs larger than chunk_size are broken further at sentence boundaries.
    3. Short units are merged greedily until the next one would exceed chunk_size.
    4. Overlap is applied by carrying the last unit of each chunk into the next.

    This avoids cutting sentences in the middle, preserving semantic coherence.
    chunk_size is a soft limit — a single sentence that exceeds it becomes its own chunk.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split on blank lines or markdown section headers."""
        paragraphs = re.split(r'\n{2,}', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_sentences(self, text: str) -> list[str]:
        """Split at sentence-ending punctuation followed by a space and uppercase."""
        # Handles Portuguese and English sentence endings.
        # Negative lookbehind avoids splitting on common abbreviations (e.g. "Sr.", "Art.").
        parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÀÃÕÂÊÔÇ\"\'\d(])', text)
        sentences = [s.strip() for s in parts if s.strip()]
        return sentences if sentences else [text.strip()]

    def _units_from(self, paragraphs: list[str]) -> list[str]:
        """Break oversized paragraphs into sentence-level units."""
        units: list[str] = []
        for para in paragraphs:
            if len(para) <= self.chunk_size:
                units.append(para)
            else:
                units.extend(self._split_sentences(para))
        return units

    def _overlap_prefix(self, current_parts: list[str]) -> list[str]:
        """Return the tail of current_parts to carry into the next chunk as overlap."""
        if not self.chunk_overlap or not current_parts:
            return []
        last = current_parts[-1]
        # Prefer carrying the full last unit if it fits within chunk_overlap budget,
        # otherwise carry a suffix of it.
        if len(last) <= self.chunk_overlap:
            return [last]
        return [last[-self.chunk_overlap:]]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []

        paragraphs = self._split_paragraphs(text)
        units = self._units_from(paragraphs)

        chunks: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for unit in units:
            separator_len = 2 if current_parts else 0  # "\n\n" between parts
            proposed_len = current_len + separator_len + len(unit)

            if proposed_len > self.chunk_size and current_parts:
                # Flush
                chunks.append("\n\n".join(current_parts))
                # Seed next chunk with overlap
                overlap = self._overlap_prefix(current_parts)
                current_parts = overlap
                current_len = sum(len(p) for p in overlap) + max(0, len(overlap) - 1) * 2

            current_parts.append(unit)
            current_len += (2 if len(current_parts) > 1 else 0) + len(unit)

        if current_parts:
            chunks.append("\n\n".join(current_parts))

        return chunks
