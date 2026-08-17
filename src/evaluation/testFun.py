import json
import math
from typing import Any


def build_chunk_id(doc: dict[str, Any]) -> str | None:
    """
    Convert retrieved document metadata into the same ID format
    used by the evaluation dataset.

    Example:
        file_id = giddiness_5b65fd85
        chunk_index = 5

    Returns:
        giddiness_5b65fd85_chunk_5
    """
    metadata = doc.get("metadata", {})

    file_id = metadata.get("file_id")
    chunk_index = metadata.get("chunk_index")

    if file_id is None or chunk_index is None:
        return None

    return f"{file_id}_chunk_{chunk_index}"

def recall_at_k(
    relevant: set[str],
    retrieved: list[str],
    k: int,
) -> float:

    if not relevant:
        return 0.0

    retrieved_k = set(retrieved[:k])

    return len(relevant & retrieved_k) / len(relevant)

def precision_at_k(
    relevant: set[str],
    retrieved: list[str],
    k: int,
) -> float:

    retrieved_k = retrieved[:k]

    relevant_retrieved = sum(
        1 for chunk_id in retrieved_k
        if chunk_id in relevant
    )

    return relevant_retrieved / k

def hit_rate_at_k(
    relevant: set[str],
    retrieved: list[str],
    k: int,
) -> float:

    retrieved_k = set(retrieved[:k])

    return 1.0 if relevant & retrieved_k else 0.0

def ndcg_at_k(
    relevant: set[str],
    retrieved: list[str],
    k: int,
) -> float:
    """
    Binary relevance:
        relevant chunk = 1
        non-relevant chunk = 0
    """

    dcg = 0.0

    for rank, chunk_id in enumerate(retrieved[:k], start=1):

        relevance = 1 if chunk_id in relevant else 0

        if relevance:
            dcg += relevance / math.log2(rank + 1)

    # Ideal ranking
    ideal_relevant_count = min(len(relevant), k)

    idcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant_count + 1)
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg

def evaluate_retrieval(
    evaluation_dataset_file: str,
    retrieval_results_file: str,
) -> dict[str, Any]:

    with open(
        evaluation_dataset_file,
        "r",
        encoding="utf-8"
    ) as f:
        evaluation_dataset = json.load(f)

    with open(
        retrieval_results_file,
        "r",
        encoding="utf-8"
    ) as f:
        retrieval_results = json.load(f)

    # Match retrieval results using query_id
    retrieval_map = {
        item["query_id"]: item
        for item in retrieval_results
    }

    ks = [1, 3, 5]

    totals = {
        k: {
            "recall": 0.0,
            "precision": 0.0,
            "hit_rate": 0.0,
            "ndcg": 0.0,
        }
        for k in ks
    }

    per_query_results = []

    for item in evaluation_dataset:

        query_id = item["query_id"]
        query = item["query"]

        relevant = set(
            item["relevant_chunk_ids"]
        )

        retrieval_item = retrieval_map.get(query_id, {})

        docs = retrieval_item.get(
            "retrieved_docs",
            []
        )

        # Support accidental nested documents
        if (
            docs
            and isinstance(docs[0], list)
        ):
            docs = docs[0]

        retrieved_chunk_ids = []

        for doc in docs:
            chunk_id = build_chunk_id(doc)

            if chunk_id is not None:
                retrieved_chunk_ids.append(chunk_id)

        query_metrics = {}

        for k in ks:

            recall = recall_at_k(
                relevant,
                retrieved_chunk_ids,
                k,
            )

            precision = precision_at_k(
                relevant,
                retrieved_chunk_ids,
                k,
            )

            hit_rate = hit_rate_at_k(
                relevant,
                retrieved_chunk_ids,
                k,
            )

            ndcg = ndcg_at_k(
                relevant,
                retrieved_chunk_ids,
                k,
            )

            query_metrics[f"recall@{k}"] = recall
            query_metrics[f"precision@{k}"] = precision
            query_metrics[f"hit_rate@{k}"] = hit_rate
            query_metrics[f"ndcg@{k}"] = ndcg

            totals[k]["recall"] += recall
            totals[k]["precision"] += precision
            totals[k]["hit_rate"] += hit_rate
            totals[k]["ndcg"] += ndcg

        per_query_results.append({
            "query_id": query_id,
            "query": query,
            "relevant_chunk_ids": list(relevant),
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "metrics": query_metrics,
        })

    total_queries = len(evaluation_dataset)

    overall_metrics = {}

    for k in ks:

        overall_metrics[f"recall@{k}"] = (
            totals[k]["recall"] / total_queries
        )

        overall_metrics[f"precision@{k}"] = (
            totals[k]["precision"] / total_queries
        )

        overall_metrics[f"hit_rate@{k}"] = (
            totals[k]["hit_rate"] / total_queries
        )

        overall_metrics[f"ndcg@{k}"] = (
            totals[k]["ndcg"] / total_queries
        )

    return {
        "total_queries": total_queries,
        "overall_metrics": overall_metrics,
        "per_query_results": per_query_results,
    }


if __name__ == '__main__' :

    results = evaluate_retrieval(
        evaluation_dataset_file="test.json",
        retrieval_results_file="retrieval_results.json",
    )

    print(json.dumps(
        results["overall_metrics"],
        indent=2
    ))