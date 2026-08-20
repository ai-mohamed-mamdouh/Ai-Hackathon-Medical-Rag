from langchain_core.documents import Document


_SCORE_FIELDS = (
    "similarity_score",
    "bm25_score",
    "rrf_score",
    "rerank_score",
)

_PROVENANCE_FIELDS = (
    "from_vector",
    "from_bm25",
    "vector_rank",
    "bm25_rank",
)


def candidate_snapshot(doc: Document, rank: int) -> dict:
    metadata = doc.metadata
    snapshot = {
        "chunk_id": metadata.get("chunk_id"),
        "file_id": metadata.get("file_id"),
        "rank": rank,
    }

    for field in _PROVENANCE_FIELDS + _SCORE_FIELDS:
        snapshot[field] = metadata.get(field)

    return snapshot


def stage_snapshot(documents: list[Document]) -> list[dict]:
    return [
        candidate_snapshot(doc, rank)
        for rank, doc in enumerate(documents, start=1)
    ]
