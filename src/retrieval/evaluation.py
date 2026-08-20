from collections.abc import Iterable


_STAGE_ORDER = (
    "vector",
    "bm25",
    "rrf",
    "after_dedup",
    "reranker_input",
    "after_rerank",
    "after_threshold",
    "final",
)


def _stage_has_gold(stage: list[dict], gold_chunk_ids: set[str]) -> bool:
    return any(
        str(candidate.get("chunk_id")) in gold_chunk_ids
        for candidate in stage
        if candidate.get("chunk_id") is not None
    )


def classify_gold_trace(
    trace: dict,
    gold_chunk_ids: Iterable[str],
) -> dict:
    """Locate where a gold chunk disappears in a debug retrieval trace."""

    gold_ids = {str(chunk_id) for chunk_id in gold_chunk_ids}
    presence = {
        stage: _stage_has_gold(trace["stages"].get(stage, []), gold_ids)
        for stage in _STAGE_ORDER
    }

    if not presence["vector"] and not presence["bm25"]:
        failure = "candidate_retrieval_failure"
    elif not presence["rrf"]:
        failure = "fusion_failure"
    elif not presence["after_dedup"]:
        failure = "deduplication_failure"
    elif not presence["reranker_input"]:
        failure = "pre_reranker_truncation"
    elif not presence["after_rerank"]:
        failure = "reranking_failure"
    elif not presence["after_threshold"]:
        failure = "threshold_failure"
    elif not presence["final"]:
        failure = "final_top_k_failure"
    else:
        failure = None

    return {
        "gold_present": presence,
        "semantic_retrieval_failure": not presence["vector"],
        "lexical_retrieval_failure": not presence["bm25"],
        "failure_category": failure,
    }
