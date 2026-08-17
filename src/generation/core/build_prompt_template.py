from langchain_core.documents import Document
from src.generation.core.prompts import RAG_SYSTEM_PROMPT
from src.retrieval.query.query import Query

def build_prompt_template(
    queries: list[Query],
    documents: list[list[Document]],
    system_prompt: str = RAG_SYSTEM_PROMPT
) -> str:

    if len(queries) != len(documents):
        raise ValueError(
            "Number of queries must match number of document groups."
        )

    if not queries:
        raise ValueError("Queries list cannot be empty.")

    original_query = queries[0].original_query

    prompt_parts = [
        f"""SYSTEM PROMPT:
{system_prompt}

ORIGINAL QUERY:
{original_query}
"""
    ]

    for query_index, (query, query_docs) in enumerate(
        zip(queries, documents),
        start=1
    ):
        question = query.normalized_query or query.original_query

        prompt_parts.append(
            f"""
QUESTION {query_index}:
{question}

CONTEXT:
"""
        )

        if not query_docs:
            prompt_parts.append("No relevant documents found.\n")
            continue

        for doc_index, doc in enumerate(query_docs, start=1):
            metadata = doc.metadata

            file_name = metadata.get("file_name", "unknown")
            page_number = metadata.get("page_number", "unknown")
            section = metadata.get("section")
            relevance_score = metadata.get("rerank_score", 0)

            metadata_lines = [
                f"File: {file_name}",
                f"Page: {page_number}",
                f"Relevance Score: {relevance_score:.4f}",
            ]

            if section:
                metadata_lines.append(
                    f"Section: {section}"
                )

            prompt_parts.append(
                f"""
[Document {doc_index}]
{chr(10).join(metadata_lines)}

Content:
{doc.page_content}
"""
            )

    return "\n".join(prompt_parts)

