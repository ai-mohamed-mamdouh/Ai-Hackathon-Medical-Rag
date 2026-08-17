from langchain_core.documents import Document
from src.generation.core.prompts import RAG_SYSTEM_PROMPT
from src.retrieval.query.query import Query

def build_prompt_template(
    queries: list[Query],
    documents: list[list[Document]],
    system_prompt: str = RAG_SYSTEM_PROMPT,
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

    # No decomposition
    if len(queries) == 1:
        prompt_parts.append("\nCONTEXT:\n")

        add_documents(
            prompt_parts=prompt_parts,
            documents=documents[0]
        )

        return "\n".join(prompt_parts)

    # With decomposition
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

        add_documents(
            prompt_parts=prompt_parts,
            documents=query_docs
        )

    return "\n".join(prompt_parts)


def add_documents(
    prompt_parts: list[str],
    documents: list[Document]
) -> None:

    if not documents:
        prompt_parts.append("No relevant documents found.")
        return

    for doc_index, doc in enumerate(documents, start=1):
        metadata = doc.metadata

        metadata_lines = [
            f"File: {metadata.get('file_name', 'unknown')}",
            f"Page: {metadata.get('page_number', 'unknown')}",
            f"Relevance Score: {metadata.get('rerank_score', 0):.4f}",
        ]

        section = metadata.get("section")

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
