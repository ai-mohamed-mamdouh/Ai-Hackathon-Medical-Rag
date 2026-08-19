import json
from pathlib import Path
from typing import Optional
from langchain_chroma import Chroma
from src.indexing.vector_store import VectorStore
from src. indexing.embeddings import Embedding

import json
from pathlib import Path
from typing import Optional, Any


def _get_chroma_collection(vector_store: Any):
    """
    Extract the underlying Chroma collection from:
    - langchain_chroma.Chroma
    - a custom VectorStore wrapper
    """

    # Case 1: Chroma object directly
    collection = getattr(vector_store, "_collection", None)

    if collection is not None and hasattr(collection, "get"):
        return collection

    # Case 2: Custom wrapper around Chroma
    possible_attributes = [
        "vector_store",
        "vectorstore",
        "store",
        "db",
        "chroma",
        "chroma_store",
    ]

    for attr_name in possible_attributes:
        inner_store = getattr(vector_store, attr_name, None)

        if inner_store is None:
            continue

        collection = getattr(inner_store, "_collection", None)

        if collection is not None and hasattr(collection, "get"):
            return collection

        # Maybe the wrapper stores the raw Chroma collection directly
        if hasattr(inner_store, "get"):
            return inner_store

    raise TypeError(
        f"Could not find Chroma collection inside "
        f"{type(vector_store).__name__}. "
        f"Available attributes: {list(vars(vector_store).keys())}"
    )


def export_file_chunks_to_json(
    vector_store,
    file_name: str,
    file_id: str,
    num_chunks: Optional[int] = None,
) -> str:

    if num_chunks is not None and num_chunks <= 0:
        raise ValueError("num_chunks must be greater than 0.")

    # Get underlying Chroma collection
    collection = _get_chroma_collection(vector_store)

    # Get all chunks for this file_id
    result = collection.get(
        where={"file_id": file_id},
        include=["documents", "metadatas"],
    )

    ids = result.get("ids", [])
    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])

    if not documents:
        raise ValueError(
            f"No chunks found for file_id: {file_id}"
        )

    chunks = []

    for chroma_id, document, metadata in zip(
        ids,
        documents,
        metadatas,
    ):
        metadata = metadata or {}

        chunk = {
            "chunk_id": metadata.get(
                "chunk_id",
                chroma_id,
            ),
            "file_id": metadata.get(
                "file_id",
                file_id,
            ),
            "version_id": metadata.get(
                "version_id",
                "",
            ),

            "file_name": metadata.get(
                "file_name",
                file_name,
            ),
            "source_type": metadata.get(
                "source_type",
                "pdf",
            ),
            "title": metadata.get(
                "title",
                "",
            ),

            "page_number": metadata.get(
                "page_number",
                0,
            ),
            "chunk_index": metadata.get(
                "chunk_index",
                0,
            ),
            "previous_chunk_index": metadata.get(
                "previous_chunk_index",
                0,
            ),
            "next_chunk_index": metadata.get(
                "next_chunk_index",
                0,
            ),

            "total_chunks": 0,

            "section": metadata.get(
                "section",
                "",
            ),
            "h1": metadata.get(
                "h1",
                "",
            ),
            "h2": metadata.get(
                "h2",
                "",
            ),
            "h3": metadata.get(
                "h3",
                "",
            ),

            "content_type": metadata.get(
                "content_type",
                "text",
            ),

            "page_content": document or "",
        }

        chunks.append(chunk)

    # Sort chunks
    chunks.sort(
        key=lambda x: x.get("chunk_index", 0)
    )

    # Total chunks belonging to the file
    total_chunks = len(chunks)

    for chunk in chunks:
        chunk["total_chunks"] = total_chunks

    # Limit if requested
    if num_chunks is not None:
        chunks = chunks[:num_chunks]

    # report.pdf -> report_chunks.json
    file_stem = Path(file_name).stem

    output_path = f"{file_stem}_chunks.json"

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved {len(chunks)} chunks")
    print(f"Total file chunks: {total_chunks}")
    print(f"Output: {output_path}")

    return output_path

if __name__ == '__main__' : 
    # file_name ="giddiness.pdf"
    # file_id="4be15b294a3626e4bb656c797a4a1a790dd8fbd4b3ca949c141919b8f414d8bb"

    file_name="WHO guidelines for malaria.pdf"
    file_id="7b89453882790218cd45734a781b96c80f8cb2d0900515d7097ae515b1a4ea21"

    embedding_model = Embedding().get_embedding_model()

    vector_store = VectorStore(embedding_model)
    export_file_chunks_to_json(vector_store=vector_store, 
                               file_id =file_id ,file_name=file_name )
    
    print(export_file_chunks_to_json)


