import json
import requests
from typing import Any


import json
import requests
from typing import Any

# Query from gpt --> my retrieval   (Q,doc)


def run_retrieval_evaluation(
    input_json_file: str,
    output_json_file: str,
    api_url: str,
) -> list[dict[str, Any]]:

    with open(input_json_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    if isinstance(queries, dict):
        queries = [queries]

    results = []

    for item in queries:
        query = item["query"]

        payload = {
            "original_query": query,
            "normalized_query": ""
        }

        response = requests.post(
            api_url,
            params={"decomposition": "false"},
            json=payload,
            timeout=60,
        )

        response.raise_for_status()

        data = response.json()

        documents = data.get("documents", [])

        if documents and isinstance(documents[0], list):
            documents = documents[0]

        results.append({
            "query_id": item["query_id"],
            "query": query,
            "relevant_chunk_ids": item["relevant_chunk_ids"],
            "retrieved_docs": documents,
        })

    with open(output_json_file, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return results


if __name__ == '__main__' :
    results = run_retrieval_evaluation(
        input_json_file="test.json",
        output_json_file="retrieval_results.json",
        api_url="http://localhost:8000/retrieval/retrieve",
    )