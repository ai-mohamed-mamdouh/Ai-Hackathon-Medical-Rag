import json
import math
import re
from pathlib import Path
from statistics import mean, median


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def percentile(values: list[float], p: float):
    """
    Calculate percentile using linear interpolation.
    p should be between 0 and 100.
    """

    if not values:
        return None

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * (p / 100)

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return values[lower]

    weight = position - lower

    return (
        values[lower] * (1 - weight)
        + values[upper] * weight
    )


def normalize_text(text: str) -> list[str]:
    """Normalize text into lowercase Unicode word tokens."""

    if not text:
        return []

    normalized = str(text).casefold()

    # Preserve the meaning of common comparison symbols before tokenization.
    normalized = normalized.replace("≥", " greater_equal ")
    normalized = normalized.replace("<=", " less_equal ")
    normalized = normalized.replace(">=", " greater_equal ")
    normalized = normalized.replace("≤", " less_equal ")
    normalized = normalized.replace(">", " greater_than ")
    normalized = normalized.replace("<", " less_than ")

    return re.findall(r"\b\w+\b", normalized, flags=re.UNICODE)


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()

    return {
        tuple(tokens[i:i + n])
        for i in range(len(tokens) - n + 1)
    }


def text_similarity(
    reference_text: str,
    candidate_text: str,
) -> float:
    """
    Evidence-oriented text similarity in [0, 1].

    This is intentionally asymmetric: it measures how much of the
    reference/gold text is covered by the retrieved chunk. That is more
    suitable for retrieval evaluation than symmetric similarity because
    a chunk can contain the complete answer plus additional context.

    Score:
        65% unigram coverage
        35% bigram coverage

    If the normalized reference appears completely inside the candidate,
    the score is 1.0.
    """

    reference_tokens = normalize_text(reference_text)
    candidate_tokens = normalize_text(candidate_text)

    if not reference_tokens or not candidate_tokens:
        return 0.0

    reference_joined = " ".join(reference_tokens)
    candidate_joined = " ".join(candidate_tokens)

    # Exact normalized containment: ideal evidence match.
    if reference_joined in candidate_joined:
        return 1.0

    reference_unigrams = set(reference_tokens)
    candidate_unigrams = set(candidate_tokens)

    unigram_coverage = (
        len(reference_unigrams & candidate_unigrams)
        / len(reference_unigrams)
    )

    reference_bigrams = _ngrams(reference_tokens, 2)
    candidate_bigrams = _ngrams(candidate_tokens, 2)

    if reference_bigrams:
        bigram_coverage = (
            len(reference_bigrams & candidate_bigrams)
            / len(reference_bigrams)
        )
    else:
        bigram_coverage = unigram_coverage

    return (
        0.65 * unigram_coverage
        + 0.35 * bigram_coverage
    )


def calculate_query_metrics(
    gold_reference_texts: list[str],
    retrieved_documents: list[dict],
    k_values=(1, 3, 5, 10),
    similarity_threshold: float = 0.75,
):
    """
    Calculate retrieval metrics for one query using text-based relevance.

    A retrieved chunk is relevant when its page_content has a text
    similarity >= similarity_threshold against at least one gold claim.

    Recall@K is claim coverage:
        covered gold claims in top-K / total gold claims
    """

    gold_texts = [
        text.strip()
        for text in gold_reference_texts
        if isinstance(text, str) and text.strip()
    ]

    document_evaluations = []

    for index, doc in enumerate(retrieved_documents, start=1):
        page_content = doc.get("page_content") or ""

        claim_scores = [
            text_similarity(
                reference_text=gold_text,
                candidate_text=page_content,
            )
            for gold_text in gold_texts
        ]

        best_score = max(claim_scores, default=0.0)

        document_evaluations.append(
            {
                "rank": doc.get("rank", index),
                "chunk_id": doc.get("chunk_id"),
                "text_similarity_score": best_score,
                "is_text_relevant": (
                    best_score >= similarity_threshold
                ),
                "claim_similarity_scores": claim_scores,
            }
        )

    metrics = {}

    # -----------------------------------------
    # Hit@K and Recall@K
    # -----------------------------------------

    for k in k_values:
        top_k_evaluations = document_evaluations[:k]

        # At least one textually relevant chunk in top-K.
        hit = int(
            any(
                item["is_text_relevant"]
                for item in top_k_evaluations
            )
        )

        # Claim-level coverage across top-K chunks.
        covered_claims = 0

        for claim_index in range(len(gold_texts)):
            best_claim_score = max(
                (
                    item["claim_similarity_scores"][claim_index]
                    for item in top_k_evaluations
                ),
                default=0.0,
            )

            if best_claim_score >= similarity_threshold:
                covered_claims += 1

        recall = (
            covered_claims / len(gold_texts)
            if gold_texts
            else 0.0
        )

        metrics[f"hit@{k}"] = hit
        metrics[f"recall@{k}"] = recall

    # -----------------------------------------
    # First Relevant Rank
    # -----------------------------------------

    first_relevant_rank = None

    for index, evaluation in enumerate(
        document_evaluations,
        start=1,
    ):
        if evaluation["is_text_relevant"]:
            first_relevant_rank = index
            break

    metrics["first_relevant_rank"] = first_relevant_rank

    # -----------------------------------------
    # Reciprocal Rank
    # -----------------------------------------

    metrics["reciprocal_rank"] = (
        1.0 / first_relevant_rank
        if first_relevant_rank is not None
        else 0.0
    )

    metrics["document_text_evaluations"] = (
        document_evaluations
    )

    return metrics

def calculate_retrieval_baseline(
    retrieval_results: list[dict],
    output_path: str | None = "baseline_retrieval_report.json",
    k_values=(1, 3, 5, 10),
    similarity_threshold: float = 0.75,
):
    """
    Input:
        list[dict] retrieval results

    Output:
        baseline_retrieval_report.json

    Text relevance:
        Retrieved page_content vs gold_claims
        Fallback: gold_answer

    Metrics:
        Hit@K
        Recall@K (gold-claim coverage)
        MRR
        First Relevant Rank
        Latency
        Empty Retrieval Rate
        API Error Rate
    """

    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError(
            "similarity_threshold must be between 0 and 1"
        )

    results = retrieval_results

    per_query = []

    successful_queries = 0
    failed_queries = 0
    empty_retrievals = 0

    latencies = []
    first_relevant_ranks = []

    # Used for global averages
    hit_values = {
        k: []
        for k in k_values
    }

    recall_values = {
        k: []
        for k in k_values
    }

    reciprocal_ranks = []

    # =========================================
    # Process each query
    # =========================================

    for result in results:

        query_id = result.get(
            "query_id"
        )

        system = result.get(
            "system",
            {},
        )

        retrieval = result.get(
            "retrieval",
            {},
        )

        gold = result.get(
            "gold",
            {},
        )

        status = system.get(
            "status"
        )

        latency_ms = system.get(
            "latency_ms"
        )

        # -------------------------------------
        # API failure
        # -------------------------------------

        if status != "success":
            failed_queries += 1

            per_query.append(
                {
                    "query_id": query_id,
                    "status": "error",
                    "error": system.get(
                        "error"
                    ),
                }
            )

            continue

        successful_queries += 1

        if latency_ms is not None:
            latencies.append(
                float(latency_ms)
            )

        # -------------------------------------
        # Extract gold text
        # -------------------------------------

        # Prefer atomic claims for retrieval evaluation.
        # Fall back to gold_answer if claims are unavailable.
        gold_claims = [
            claim
            for claim in gold.get(
                "gold_claims",
                [],
            )
            if isinstance(claim, str)
            and claim.strip()
        ]

        gold_answer = gold.get(
            "gold_answer"
        )

        gold_reference_texts = (
            gold_claims
            if gold_claims
            else (
                [gold_answer]
                if isinstance(gold_answer, str)
                and gold_answer.strip()
                else []
            )
        )

        # Keep IDs only for auditing/debugging. They are NOT used
        # to decide relevance anymore.
        gold_chunk_ids = gold.get(
            "gold_chunk_ids",
            [],
        )

        if not gold_chunk_ids:
            gold_chunk_ids = [
                evidence.get("chunk_id")
                for evidence in gold.get(
                    "gold_evidence",
                    [],
                )
                if evidence.get("chunk_id")
            ]

        # -------------------------------------
        # Extract retrieved docs
        # -------------------------------------

        documents = retrieval.get(
            "documents",
            [],
        )

        if not documents:
            empty_retrievals += 1

        # -------------------------------------
        # Calculate query metrics
        # -------------------------------------

        metrics = calculate_query_metrics(
            gold_reference_texts=gold_reference_texts,
            retrieved_documents=documents,
            k_values=k_values,
            similarity_threshold=similarity_threshold,
        )

        # Collect metrics
        for k in k_values:
            hit_values[k].append(
                metrics[f"hit@{k}"]
            )

            recall_values[k].append(
                metrics[f"recall@{k}"]
            )

        reciprocal_ranks.append(
            metrics[
                "reciprocal_rank"
            ]
        )

        first_rank = metrics[
            "first_relevant_rank"
        ]

        if first_rank is not None:
            first_relevant_ranks.append(
                first_rank
            )

        # -------------------------------------
        # Save per-query result
        # -------------------------------------

        per_query.append(
            {
                "query_id": query_id,

                "original_query": (
                    result
                    .get("input", {})
                    .get("original_query")
                ),

                "gold_reference_texts": (
                    gold_reference_texts
                ),

                "gold_chunk_ids_for_audit": (
                    gold_chunk_ids
                ),

                "similarity_threshold": (
                    similarity_threshold
                ),

                "retrieved_chunk_ids": [
                    doc.get(
                        "chunk_id"
                    )
                    for doc in documents
                ],

                "retrieved_count": len(
                    documents
                ),

                **metrics,

                "latency_ms": latency_ms,

                "status": "success",
            }
        )

    # =========================================
    # Aggregate Metrics
    # =========================================

    aggregate = {}

    # -----------------------------------------
    # Hit@K
    # -----------------------------------------

    for k in k_values:
        values = hit_values[k]

        aggregate[
            f"hit@{k}"
        ] = (
            mean(values)
            if values
            else 0.0
        )

    # -----------------------------------------
    # Recall@K
    # -----------------------------------------

    for k in k_values:
        values = recall_values[k]

        aggregate[
            f"recall@{k}"
        ] = (
            mean(values)
            if values
            else 0.0
        )

    # -----------------------------------------
    # MRR
    # -----------------------------------------

    aggregate["mrr"] = (
        mean(reciprocal_ranks)
        if reciprocal_ranks
        else 0.0
    )

    # -----------------------------------------
    # First Relevant Rank
    # -----------------------------------------

    aggregate[
        "mean_first_relevant_rank"
    ] = (
        mean(first_relevant_ranks)
        if first_relevant_ranks
        else None
    )

    aggregate[
        "median_first_relevant_rank"
    ] = (
        median(first_relevant_ranks)
        if first_relevant_ranks
        else None
    )

    # -----------------------------------------
    # Retrieval success
    # -----------------------------------------

    total_queries = len(results)

    aggregate[
        "empty_retrieval_rate"
    ] = (
        empty_retrievals
        / successful_queries
        if successful_queries
        else 0.0
    )

    aggregate[
        "api_error_rate"
    ] = (
        failed_queries
        / total_queries
        if total_queries
        else 0.0
    )

    # -----------------------------------------
    # Latency
    # -----------------------------------------

    latency_stats = {
        "mean_ms": (
            mean(latencies)
            if latencies
            else None
        ),

        "p50_ms": percentile(
            latencies,
            50,
        ),

        "p95_ms": percentile(
            latencies,
            95,
        ),

        "p99_ms": percentile(
            latencies,
            99,
        ),

        "min_ms": (
            min(latencies)
            if latencies
            else None
        ),

        "max_ms": (
            max(latencies)
            if latencies
            else None
        ),
    }

    # =========================================
    # Final Report
    # =========================================

    report = {
        "evaluation": {
            "relevance_method": "text_based_similarity",
            "gold_text_source": "gold_claims_fallback_gold_answer",
            "similarity_threshold": similarity_threshold,
            "similarity_formula": (
                "exact containment => 1.0; otherwise "
                "0.65 * unigram_coverage + 0.35 * bigram_coverage"
            ),
        },

        "summary": {
            "total_queries": (
                total_queries
            ),

            "successful_queries": (
                successful_queries
            ),

            "failed_queries": (
                failed_queries
            ),

            "empty_retrievals": (
                empty_retrievals
            ),
        },

        "metrics": aggregate,

        "latency": latency_stats,

        "per_query": per_query,
    }

    if output_path:
        save_json(
            report,
            output_path,
        )

    return report


if __name__ == '__main__' :
    retrieval_results = load_json(
        "evaluation/test_dataset/malaria_retrieval_raw_results.json"
    )

    report = calculate_retrieval_baseline(
        retrieval_results=retrieval_results,
        output_path="baseline_retrieval_report.json",
        similarity_threshold=0.75,
    )

    print(
        json.dumps(
            report["metrics"],
            indent=2,
        )
    )

    print(
        json.dumps(
            report["latency"],
            indent=2,
        )
    )