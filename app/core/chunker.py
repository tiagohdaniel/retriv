class TextChunker:
    """Splits text into overlapping chunks for indexing.

    Overlap ensures that sentences split across chunk boundaries
    are still retrievable from either chunk.
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
