import hashlib
from collections import defaultdict
from langchain_core.documents import Document
from src.config.settings import settings


class RRFFusion:

    def __init__(self, rrf_k: int = settings.RRF_K):
        self.rrf_k = rrf_k

    def fuse(
        self,
        ranked_lists: list[list[Document]],
        top_k: int | None = None
    ) -> list[Document]:

        scores = defaultdict(float)
        documents = {}

        for docs in ranked_lists:

            for rank, doc in enumerate(docs, start=1):

                doc_id = self._get_doc_id(doc)

                # RRF score
                scores[doc_id] += 1 / (self.rrf_k + rank)

                # Keep one copy of the document
                if doc_id not in documents:
                    documents[doc_id] = doc

        # Sort documents by RRF score
        ranked_doc_ids = sorted(
            scores,
            key=scores.get,
            reverse=True
        )

        results = []

        for doc_id in ranked_doc_ids:

            doc = documents[doc_id]

            # Save RRF score in metadata
            doc.metadata["rrf_score"] = scores[doc_id]

            results.append(doc)

        if top_k is not None:
            return results[:top_k]

        return results

    @staticmethod
    def _get_doc_id(doc: Document) -> str:

        # Use chunk_id if available
        chunk_id = doc.metadata.get("chunk_id")

        if chunk_id:
            return str(chunk_id)

        # Fallback deterministic ID
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page_number", "")

        raw_id = f"{source}:{page}:{doc.page_content}"

        return hashlib.sha256(
            raw_id.encode("utf-8")
        ).hexdigest()
    