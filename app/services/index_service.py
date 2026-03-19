import re

from app.schemas.models import IndexRequest, IndexResponse
from app.core.logging_config import get_logger
from app.core.metrics import documents_indexed_total, chunks_indexed_total

logger = get_logger("retriv.index")

_EMBED_BATCH_SIZE = 32  # chunks per ONNX inference call — prevents OOM on large documents

# Minimum chunk length to index. Shorter chunks are headers, footers, or navigation
# fragments that produce false matches without adding retrieval value.
_MIN_CHUNK_CHARS = 80

# PDF navigation and header noise patterns
_FOOTER_RE = re.compile(r'Voltar ao Sum[aá]rio\s*\d*\s*\n?', re.IGNORECASE)
_HYPHEN_RE = re.compile(r'(\w)-\n(\w)')          # broken hyphenation across lines
_DOTS_LINE_RE = re.compile(r'^[.\s]{5,}\d*\s*$', re.MULTILINE)  # ".......38"
_MULTI_SPACE_RE = re.compile(r'[ \t]{2,}')
_MULTI_NL_RE = re.compile(r'\n{3,}')

# TOC line pattern: any line ending with 3+ dots optionally followed by a page number
_TOC_LINE_RE = re.compile(r'\.{3,}\s*\d*\s*$')


def _is_toc_chunk(chunk: str) -> bool:
    """Return True if the chunk is a table-of-contents entry.

    Must be called on the RAW chunk BEFORE cleaning — _clean_pdf_text strips
    the trailing dots from TOC lines, making them undetectable afterwards.

    Heuristic: ≥30% of non-empty lines end with the pattern "....N"
    (dots followed by a page number), which is the standard PDF TOC format.
    These chunks match queries but contain no actual content — they only
    list section names and page numbers.
    """
    lines = [l for l in chunk.splitlines() if l.strip()]
    if not lines:
        return False
    toc_lines = sum(1 for l in lines if _TOC_LINE_RE.search(l))
    return toc_lines / len(lines) >= 0.3


def _clean_pdf_text(text: str) -> str:
    """Remove PDF navigation artifacts before chunking.

    Targets structural noise common in Brazilian government PDFs:
    - "Voltar ao Sumário N" navigation links at page bottoms
    - Broken hyphenation from line-wrapping ("decla-\\nração" → "declaração")
    - Index/TOC lines that are just dots ("..............38")
    - Excess whitespace
    """
    text = _FOOTER_RE.sub('', text)
    text = _HYPHEN_RE.sub(r'\1\2', text)
    text = _DOTS_LINE_RE.sub('', text)
    text = _MULTI_SPACE_RE.sub(' ', text)
    text = _MULTI_NL_RE.sub('\n\n', text)
    return text.strip()


class IndexService:
    """Orchestrates: chunk → embed → store.

    Re-indexing an existing source_id replaces all its chunks.
    """

    def __init__(self, chunker, embedding_service, vector_store):
        self.chunker = chunker
        self.embedding = embedding_service
        self.vector_store = vector_store

    def index(self, request: IndexRequest, tenant_id: str | None = None) -> IndexResponse:
        # delete existing chunks first — re-indexing the same source_id is idempotent
        self.vector_store.delete_source(request.source_id, tenant_id=tenant_id)

        # Single-pass: chunk raw → detect TOC on raw chunk → clean each chunk → filter.
        # This avoids the index alignment risk of double-chunking (raw for detection,
        # clean for embedding). Both detection and cleaning operate on the same unit.
        raw_chunks = self.chunker.chunk(request.content)
        before = len(raw_chunks)
        chunks = []
        for raw_chunk in raw_chunks:
            if _is_toc_chunk(raw_chunk):
                continue
            cleaned = _clean_pdf_text(raw_chunk)
            if len(cleaned.strip()) >= _MIN_CHUNK_CHARS:
                chunks.append(cleaned)
        filtered = before - len(chunks)
        if filtered:
            logger.debug("index_chunks_filtered", source_id=request.source_id, filtered=filtered)

        if not chunks:
            logger.warning("index_no_chunks", source_id=request.source_id)
            return IndexResponse(source_id=request.source_id, chunks_indexed=0)

        embeddings: list[list[float]] = []
        for i in range(0, len(chunks), _EMBED_BATCH_SIZE):
            embeddings.extend(self.embedding.encode(chunks[i : i + _EMBED_BATCH_SIZE], task="document"))

        self.vector_store.upsert_chunks(
            source_id=request.source_id,
            title=request.title,
            chunks=chunks,
            embeddings=embeddings,
            tenant_id=tenant_id,
            extra_metadata=request.metadata,
        )

        documents_indexed_total.inc()
        chunks_indexed_total.inc(len(chunks))
        logger.info(
            "index_completed",
            source_id=request.source_id,
            chunks_indexed=len(chunks),
            tenant_id=tenant_id,
        )
        return IndexResponse(source_id=request.source_id, chunks_indexed=len(chunks))
