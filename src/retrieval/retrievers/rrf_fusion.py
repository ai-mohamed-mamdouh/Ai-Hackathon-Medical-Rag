from collections import defaultdict

from langchain_core.documents import Document

from src.config.settings import settings
from src.retrieval.retrievers.document_id import get_document_id


class RRFFusion:
    def __init__(self, rrf_k: int = settings.RRF_K):
        self.rrf_k = rrf_k

    def fuse(
        self,
        ranked_lists: list[list[Document]],
        top_k: int | None = None,
    ) -> list[Document]:
        scores = defaultdict(float)
        documents: dict[str, Document] = {}

        for docs in ranked_lists:
            seen_in_list = set()

            for rank, doc in enumerate(docs, start=1):
                doc_id = get_document_id(doc)

                # A retriever should contribute at most once per candidate.
                # Otherwise duplicate rows inside one branch get extra RRF votes.
                if doc_id in seen_in_list:
                    continue
                seen_in_list.add(doc_id)

                scores[doc_id] += 1 / (self.rrf_k + rank)
                if doc_id not in documents:
                    documents[doc_id] = Document(
                        page_content=doc.page_content,
                        metadata=dict(doc.metadata),
                    )
                else:
                    self._merge_metadata(documents[doc_id], doc)

        # Preserve the legacy stable tie behavior. Tie-breaking itself should
        # be evaluated separately rather than changed without benchmark data.
        ranked_doc_ids = sorted(
            scores,
            key=scores.get,
            reverse=True,
        )

        results = []
        for doc_id in ranked_doc_ids:
            doc = documents[doc_id]
            doc.metadata["rrf_score"] = float(scores[doc_id])
            results.append(doc)

        if top_k is not None:
            return results[:top_k]
        return results

    @staticmethod
    def _merge_metadata(target: Document, source: Document) -> None:
        for field in (
            "from_vector",
            "from_bm25",
            "vector_rank",
            "bm25_rank",
            "similarity_score",
            "bm25_score",
        ):
            value = source.metadata.get(field)
            if value is not None:
                target.metadata[field] = value
