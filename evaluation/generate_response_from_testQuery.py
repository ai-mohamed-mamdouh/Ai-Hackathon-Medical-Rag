import json
import time
from pathlib import Path

import requests


DEFAULT_ENDPOINT = (
    "http://127.0.0.1:8000/retrieval/retrieve?decomposition=false"
)


def load_json_list(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must contain a list.")

    return data


def extract_retrieved_documents(response_data: dict) -> list[dict]:
    """
    Expected API shape:

    {
        "documents": [
            [
                {
                    "metadata": {...},
                    "page_content": "...",
                    "type": "Document"
                }
            ]
        ],
        "queries": [...]
    }
    """

    documents_wrapper = response_data.get("documents", [])

    if not documents_wrapper:
        return []

    # API returns one query, so documents[0] belongs to that query.
    documents = documents_wrapper[0]

    if not isinstance(documents, list):
        return []

    extracted = []

    for rank, doc in enumerate(documents, start=1):
        if not isinstance(doc, dict):
            continue

        metadata = doc.get("metadata") or {}

        extracted.append(
            {
                "rank": rank,

                # Identity
                "chunk_id": metadata.get("chunk_id"),
                "file_id": metadata.get("file_id"),
                "version_id": metadata.get("version_id"),
                "file_name": metadata.get("file_name"),

                # Location
                "page_number": metadata.get("page_number"),
                "chunk_index": metadata.get("chunk_index"),
                "section": metadata.get("section"),

                # Retrieval scores
                "similarity_score": metadata.get("similarity_score"),
                "bm25_score": metadata.get("bm25_score"),
                "rrf_score": metadata.get("rrf_score"),
                "rerank_score": metadata.get("rerank_score"),

                # Useful later for debugging / evaluation
                "page_content": doc.get("page_content", ""),
            }
        )

    return extracted


def run_retrieval_evaluation(
    test_dataset_path: str,
    output_path: str,
    endpoint: str = DEFAULT_ENDPOINT,
    timeout: float = 60.0,
) -> dict:
    """
    Run every query in the clean retrieval test dataset against
    the Retrieval API and save raw results.

    This function DOES NOT calculate metrics.
    """

    test_cases = load_json_list(test_dataset_path)

    results = []

    session = requests.Session()
    session.headers.update(
        {
            "accept": "application/json",
            "Content-Type": "application/json",
        }
    )

    for index, test_case in enumerate(test_cases, start=1):
        query_id = test_case.get("query_id")

        original_query = test_case.get(
            "original_query",
            "",
        ).strip()

        normalized_query = test_case.get(
            "normalized_query",
            "",
        ).strip()

        gold_evidence = test_case.get(
            "gold_evidence",
            [],
        )

        gold_chunk_ids = [
            evidence.get("chunk_id")
            for evidence in gold_evidence
            if evidence.get("chunk_id")
        ]

        payload = {
            "original_query": original_query,
            "normalized_query": normalized_query,
        }

        start_time = time.perf_counter()

        try:
            response = session.post(
                endpoint,
                json=payload,
                timeout=timeout,
            )

            latency_ms = (
                time.perf_counter() - start_time
            ) * 1000

            response.raise_for_status()

            response_data = response.json()

            retrieved_documents = (
                extract_retrieved_documents(
                    response_data
                )
            )

            returned_queries = response_data.get(
                "queries",
                [],
            )

            result = {
                "query_id": query_id,

                "input": {
                    "original_query": original_query,
                    "normalized_query": normalized_query,
                },

                "gold": {
                    "gold_answer": test_case.get(
                        "gold_answer"
                    ),
                    "gold_claims": test_case.get(
                        "gold_claims",
                        [],
                    ),
                    "gold_evidence": gold_evidence,
                    "gold_chunk_ids": gold_chunk_ids,
                },

                "retrieval": {
                    "returned_queries": returned_queries,
                    "documents": retrieved_documents,
                    "retrieved_count": len(
                        retrieved_documents
                    ),
                },

                "system": {
                    "status": "success",
                    "status_code": response.status_code,
                    "latency_ms": round(
                        latency_ms,
                        3,
                    ),
                    "error": None,
                },
            }

        except requests.RequestException as exc:
            latency_ms = (
                time.perf_counter() - start_time
            ) * 1000

            result = {
                "query_id": query_id,

                "input": {
                    "original_query": original_query,
                    "normalized_query": normalized_query,
                },

                "gold": {
                    "gold_answer": test_case.get(
                        "gold_answer"
                    ),
                    "gold_claims": test_case.get(
                        "gold_claims",
                        [],
                    ),
                    "gold_evidence": gold_evidence,
                    "gold_chunk_ids": gold_chunk_ids,
                },

                "retrieval": {
                    "returned_queries": [],
                    "documents": [],
                    "retrieved_count": 0,
                },

                "system": {
                    "status": "error",
                    "status_code": getattr(
                        getattr(
                            exc,
                            "response",
                            None,
                        ),
                        "status_code",
                        None,
                    ),
                    "latency_ms": round(
                        latency_ms,
                        3,
                    ),
                    "error": str(exc),
                },
            }

        except (ValueError, TypeError) as exc:
            latency_ms = (
                time.perf_counter() - start_time
            ) * 1000

            result = {
                "query_id": query_id,

                "input": {
                    "original_query": original_query,
                    "normalized_query": normalized_query,
                },

                "gold": {
                    "gold_answer": test_case.get(
                        "gold_answer"
                    ),
                    "gold_claims": test_case.get(
                        "gold_claims",
                        [],
                    ),
                    "gold_evidence": gold_evidence,
                    "gold_chunk_ids": gold_chunk_ids,
                },

                "retrieval": {
                    "returned_queries": [],
                    "documents": [],
                    "retrieved_count": 0,
                },

                "system": {
                    "status": "error",
                    "status_code": None,
                    "latency_ms": round(
                        latency_ms,
                        3,
                    ),
                    "error": (
                        f"Invalid API response: {exc}"
                    ),
                },
            }

        results.append(result)

        print(
            f"[{index}/{len(test_cases)}] "
            f"{query_id} -> "
            f"{result['system']['status']} -> "
            f"{result['retrieval']['retrieved_count']} docs -> "
            f"{result['system']['latency_ms']} ms"
        )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    successful = sum(
        1
        for result in results
        if result["system"]["status"] == "success"
    )

    failed = len(results) - successful

    summary = {
        "total_queries": len(results),
        "successful_queries": successful,
        "failed_queries": failed,
        "output_file": str(output_path),
    }

    return summary


if __name__ == '__main__' :
    summary = run_retrieval_evaluation(
    test_dataset_path="evaluation/test_dataset/guideline_test_dataset_clean.json",
    output_path="evaluation/test_dataset/retrieval_raw_results.json",
    )

    print(summary)