import json
import math
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


def calculate_query_metrics(
    gold_chunk_ids: list[str],
    retrieved_documents: list[dict],
    k_values=(1, 3, 5, 10),
):
    """
    Calculate retrieval metrics for one query.
    """

    gold_set = {
        chunk_id
        for chunk_id in gold_chunk_ids
        if chunk_id
    }

    retrieved_ids = [
        doc.get("chunk_id")
        for doc in retrieved_documents
        if doc.get("chunk_id")
    ]

    metrics = {}

    # -----------------------------------------
    # Hit@K and Recall@K
    # -----------------------------------------

    for k in k_values:
        top_k = retrieved_ids[:k]

        relevant_found = gold_set.intersection(
            top_k
        )

        # Did we retrieve at least one gold chunk?
        hit = 1 if relevant_found else 0

        # How much of all gold evidence was retrieved?
        recall = (
            len(relevant_found) / len(gold_set)
            if gold_set
            else 0.0
        )

        metrics[f"hit@{k}"] = hit
        metrics[f"recall@{k}"] = recall

    # -----------------------------------------
    # First Relevant Rank
    # -----------------------------------------

    first_relevant_rank = None

    for rank, chunk_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if chunk_id in gold_set:
            first_relevant_rank = rank
            break

    metrics["first_relevant_rank"] = (
        first_relevant_rank
    )

    # -----------------------------------------
    # Reciprocal Rank
    # -----------------------------------------

    if first_relevant_rank is None:
        reciprocal_rank = 0.0
    else:
        reciprocal_rank = (
            1.0 / first_relevant_rank
        )

    metrics["reciprocal_rank"] = (
        reciprocal_rank
    )

    return metrics


def calculate_retrieval_baseline(
    retrieval_results_path: str,
    output_path: str = "baseline_retrieval_report.json",
    k_values=(1, 3, 5, 10),
):
    """
    Input:
        retrieval_raw_results.json

    Output:
        baseline_retrieval_report.json

    Metrics:
        Hit@K
        Recall@K
        MRR
        First Relevant Rank
        Latency
        Empty Retrieval Rate
        API Error Rate
    """

    results = load_json(
        retrieval_results_path
    )

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
        # Extract gold
        # -------------------------------------

        gold_chunk_ids = gold.get(
            "gold_chunk_ids",
            [],
        )

        # Fallback if gold_chunk_ids missing
        if not gold_chunk_ids:
            gold_chunk_ids = [
                evidence.get(
                    "chunk_id"
                )
                for evidence in gold.get(
                    "gold_evidence",
                    [],
                )
                if evidence.get(
                    "chunk_id"
                )
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
            gold_chunk_ids=gold_chunk_ids,
            retrieved_documents=documents,
            k_values=k_values,
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

                "gold_chunk_ids": (
                    gold_chunk_ids
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

    save_json(
        report,
        output_path,
    )

    return report


if __name__ == '__main__' :
    report = calculate_retrieval_baseline(
        retrieval_results_path="evaluation/test_dataset/retrieval_raw_results.json",
        output_path="baseline_retrieval_report.json",
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