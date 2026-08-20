from dataclasses import dataclass

from src.config.settings import settings


@dataclass(frozen=True)
class RetrievalConfig:
    vector_candidate_k: int
    bm25_candidate_k: int

    reranker_input_k: int
    reranker_output_k: int

    final_top_k: int

    rrf_k: int
    relevance_threshold: float | None
    semantic_dedup_threshold: float 
    lexical_dedup_threshold: float

    bm25_min_score: float = 0.0
    parallel_retrieval: bool = False
    reranker_batch_size: int | None = None

    @classmethod
    def from_settings(cls) -> "RetrievalConfig":
        legacy_top_k = settings.TOP_K

        return cls(
            vector_candidate_k=getattr(
                settings,
                "VECTOR_CANDIDATE_K",
                legacy_top_k,
            ),
            lexical_dedup_threshold=getattr(
                settings,
                "LEXICAL_DEDUP_THRESHOLD",
                0.60,
            ),
            bm25_candidate_k=getattr(
                settings,
                "BM25_CANDIDATE_K",
                legacy_top_k,
            ),
            reranker_input_k=getattr(
                settings,
                "RERANKER_INPUT_K",
                legacy_top_k,
            ),
            reranker_output_k=getattr(
                settings,
                "RERANKER_OUTPUT_K",
                legacy_top_k,
            ),
            final_top_k=getattr(
                settings,
                "FINAL_TOP_K",
                legacy_top_k,
            ),
            rrf_k=settings.RRF_K,
            relevance_threshold=settings.RELEVANCE_THRESHOLD,
            semantic_dedup_threshold=getattr(
                settings,
                "SEMANTIC_DEDUP_THRESHOLD",
                0.94,
            ),
            bm25_min_score=getattr(
                settings,
                "BM25_MIN_SCORE",
                0.0,
            ),
            parallel_retrieval=getattr(
                settings,
                "PARALLEL_RETRIEVAL",
                False,
            ),
            reranker_batch_size=getattr(
                settings,
                "RERANKER_BATCH_SIZE",
                None,
            ),
        )