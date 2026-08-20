import hashlib

from langchain_core.documents import Document


def get_document_id(doc: Document) -> str:
    """Return a deterministic candidate identity shared by all retrieval stages."""

    chunk_id = doc.metadata.get("chunk_id")
    if chunk_id is not None and str(chunk_id) != "":
        return f"chunk:{chunk_id}"

    file_id = doc.metadata.get("file_id")
    chunk_index = doc.metadata.get("chunk_index")
    if file_id is not None and chunk_index is not None:
        return f"file_chunk:{file_id}:{chunk_index}"

    source = doc.metadata.get("source", "")
    page = doc.metadata.get("page_number", "")
    raw_id = f"{source}:{page}:{doc.page_content}"
    digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()
    return f"fallback:{digest}"
