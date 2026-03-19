import re
from typing import NamedTuple

# Abbreviations that must never be treated as sentence boundaries.
# Used by SemanticChunker._split_sentences via a placeholder substitution:
# the period in each match is temporarily replaced with \x00 before sentence
# splitting and restored afterwards.
#
# Covers legal/fiscal Portuguese:
#   art / arts / dec / res / port / lei / ord  — legal references
#   inc / al / cap / sec / par / tit           — document structure
#   no / nº / nr / n                           — numbering
#   cf / op / cit / ibid / apud                — citations
#   sr / sra / dr / dra / prof / eng / adv     — titles/treatment
_ABBREV_RE = re.compile(
    r'\b(arts?|inc|al|cap|sec|par|tit|vol|p[aá]g?|dec|res|port|lei|ord'
    r'|n[oº°r]?|cf|op|cit|ibid|apud|sr[a]?|dr[a]?|prof|eng|adv|proc)\.'
    r'(?=\s)',
    re.IGNORECASE,
)


class _Unit(NamedTuple):
    """Internal representation of a text unit produced during chunking.

    ``hard`` is True only for units produced by ``_hard_split`` — word-boundary
    cuts on content without sentence structure (tables, legal enumerations, PDFs
    with missing punctuation). Hard units receive a char-suffix overlap fallback
    instead of the full-unit semantic overlap used for normal sentences.
    """

    text: str
    hard: bool


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
    3. Units still exceeding chunk_size (e.g. table rows without punctuation) are
       hard-split at word boundaries as a last resort — chunk_size is a hard upper bound.
    4. Short units are merged greedily until the next one would exceed chunk_size.
    5. Overlap is applied by carrying complete semantic units from the tail of each
       chunk into the next. Hard-split units (non-semantic content) fall back to a
       char-suffix overlap instead.

    This avoids cutting sentences in the middle, preserving semantic coherence.
    No chunk will ever exceed chunk_size characters, preventing OOM in downstream
    embedding models when processing tabular PDFs or other non-prose content.
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
        """Split at sentence-ending punctuation followed by a space and uppercase or digit.

        Abbreviations common in legal/fiscal Portuguese (art., inc., nº., dec., …)
        are protected via a placeholder substitution: their periods are temporarily
        replaced with \\x00 before splitting and restored afterwards. This prevents
        false splits on "art. 966", "inc. I", "cap. II", etc.
        """
        protected = _ABBREV_RE.sub(lambda m: m.group().replace('.', '\x00'), text)
        parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÀÃÕÂÊÔÇ\"\'\d(])', protected)
        sentences = [s.replace('\x00', '.').strip() for s in parts if s.strip()]
        return sentences if sentences else [text.strip()]

    def _units_from(self, paragraphs: list[str]) -> list[_Unit]:
        """Break oversized paragraphs into sentence-level units.

        Falls back to hard word-boundary splitting for content without sentence
        endings (e.g. table rows, legal codes, tabular PDFs). This guarantees
        every unit is at most chunk_size characters, preventing OOM in the
        embedding model.
        """
        units: list[_Unit] = []
        for para in paragraphs:
            if len(para) <= self.chunk_size:
                units.append(_Unit(para, hard=False))
            else:
                for sentence in self._split_sentences(para):
                    if len(sentence) <= self.chunk_size:
                        units.append(_Unit(sentence, hard=False))
                    else:
                        units.extend(self._hard_split(sentence))
        return units

    def _hard_split(self, text: str) -> list[_Unit]:
        """Split text that has no natural boundaries at word boundaries.

        Last-resort fallback for tabular/code content where neither paragraph
        nor sentence splitting produces units within chunk_size. Splits at the
        last space before chunk_size, falling back to a hard character cut if
        no space is found in the second half of the window.

        All produced units are marked ``hard=True`` so that ``_overlap_prefix``
        can apply a char-suffix fallback instead of the semantic full-unit strategy.
        """
        parts: list[_Unit] = []
        while len(text) > self.chunk_size:
            boundary = text[: self.chunk_size].rfind(" ")
            if boundary < self.chunk_size // 2:
                boundary = self.chunk_size  # no good word boundary — hard cut
            parts.append(_Unit(text[:boundary].strip(), hard=True))
            text = text[boundary:].strip()
        if text:
            parts.append(_Unit(text, hard=True))
        return parts

    def _is_section_header(self, unit: str) -> bool:
        """Return True if unit starts with a numbered section heading.

        Matches patterns like "1.5. CONCEITOS" or "2.3.1. DEFINIÇÃO" —
        common in Brazilian government and legal documents. These are strong
        semantic boundaries that should never be merged with preceding content.
        """
        return bool(re.match(r'^\d{1,2}(\.\d{1,2}){1,3}\.?\s+[A-ZÁÉÍÓÚÀÃÕÂÊÔÇ]{2}', unit))

    def _overlap_prefix(self, current_parts: list[_Unit]) -> list[_Unit]:
        """Return units from the tail of the current chunk to seed the next one.

        Semantic units (hard=False):
            Always carried whole — never truncated. The last unit is always
            included, even if it exceeds the chunk_overlap budget, because a
            complete sentence is always preferable to a raw character fragment.
            Additional units are prepended (working backwards) as long as they
            fit within the remaining budget.

        Hard-split units (hard=True):
            Non-semantic content (tables, enumerations). The last unit receives
            a char-suffix fallback (last chunk_overlap chars) if it is the only
            candidate and nothing semantic was found closer to the tail.
        """
        if not self.chunk_overlap or not current_parts:
            return []

        result: list[_Unit] = []
        budget = self.chunk_overlap

        for unit in reversed(current_parts):
            if unit.hard:
                # Non-semantic unit: char-suffix fallback, only when nothing
                # better has been found yet (i.e. the tail of the chunk is
                # hard-split content with no trailing semantic sentence).
                if not result:
                    suffix = unit.text[-budget:] if len(unit.text) > budget else unit.text
                    result.insert(0, _Unit(suffix, hard=True))
                break

            # Semantic unit: always carry the first one whole (mandatory),
            # then continue backwards while budget allows.
            if not result or len(unit.text) <= budget:
                result.insert(0, unit)
                budget -= len(unit.text)
                if budget <= 0:
                    break
            else:
                break

        return result

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
        current_parts: list[_Unit] = []
        current_len = 0

        for unit in units:
            separator_len = 2 if current_parts else 0  # "\n\n" between parts
            proposed_len = current_len + separator_len + len(unit.text)

            # Section headers are hard boundaries — never merge with preceding content
            force_break = self._is_section_header(unit.text) and current_parts

            if (proposed_len > self.chunk_size or force_break) and current_parts:
                # Flush
                chunks.append("\n\n".join(p.text for p in current_parts))
                if force_break:
                    # Section header starts a fresh chunk — no overlap from the
                    # previous section. Carrying context across section boundaries
                    # splits the header from its content and pollutes retrieval.
                    current_parts = []
                    current_len = 0
                else:
                    # Seed next chunk with overlap
                    overlap = self._overlap_prefix(current_parts)
                    current_parts = overlap
                    current_len = sum(len(p.text) for p in overlap) + max(0, len(overlap) - 1) * 2

                    # Guard: if the overlap alone leaves no room for the incoming unit,
                    # the next flush would produce an oversized chunk. Resolve now:
                    if current_parts and current_len + 2 + len(unit.text) > self.chunk_size:
                        if any(not p.hard for p in current_parts):
                            chunks.append("\n\n".join(p.text for p in current_parts))
                        current_parts = []
                        current_len = 0

            current_parts.append(unit)
            current_len += (2 if len(current_parts) > 1 else 0) + len(unit.text)

        if current_parts:
            chunks.append("\n\n".join(p.text for p in current_parts))

        return chunks
