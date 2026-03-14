import re

_RRF_K = 60  # standard constant — controls rank penalty steepness


class BM25Searcher:
    """BM25 keyword search backed by rank-bm25 (pure Python).

    BM25 (Okapi BM25) scores documents by term frequency and inverse document
    frequency. Unlike cosine similarity, it rewards exact keyword matches and
    penalises very long documents — complementary to dense vector search.

    Falls back to empty results gracefully if rank-bm25 is not installed,
    so the caller degrades to semantic-only without crashing.
    """

    def search(self, query: str, corpus: list[dict], top_k: int) -> list[dict]:
        if not corpus:
            return []
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return []

        tokenized_corpus = [self._tokenize(doc["document"]) for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(self._tokenize(query))

        scored = [
            (score, doc)
            for score, doc in zip(scores, corpus)
            if score > 0
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

    def _tokenize(self, text: str) -> list[str]:
        """Lower-case word tokeniser — works for Portuguese and English."""
        return re.findall(r'\w+', text.lower())


def rrf_merge(
    semantic_results: list[dict],
    bm25_results: list[dict],
    k: int = _RRF_K,
) -> list[dict]:
    """Merge two ranked lists with Reciprocal Rank Fusion (RRF).

    score(d) = Σ 1/(k + rank_i(d))   — higher is better.

    Uses doc["id"] as the merge key. Semantic results take metadata
    precedence (they carry the ChromaDB distance field). BM25-only docs
    receive distance=0.4 as a neutral placeholder within the typical
    acceptance threshold.
    """
    rrf_scores: dict[str, float] = {}
    docs_by_id: dict[str, dict] = {}

    for rank, doc in enumerate(semantic_results, 1):
        doc_id = doc["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        docs_by_id[doc_id] = doc  # canonical: has ChromaDB distance

    for rank, doc in enumerate(bm25_results, 1):
        doc_id = doc["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        if doc_id not in docs_by_id:
            docs_by_id[doc_id] = {**doc, "distance": 0.4}

    sorted_ids = sorted(rrf_scores, key=lambda d: rrf_scores[d], reverse=True)
    return [docs_by_id[doc_id] for doc_id in sorted_ids]
