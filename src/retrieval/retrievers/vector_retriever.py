from langchain_core.documents import Document

from src.indexing.vector_store import VectorStore
from src.config.settings import settings


class VectorRetriever:
    def __init__(self, vector_store: VectorStore, k: int | None = None):
        self.vector_store = vector_store.get_vector_store()
        self.k = settings.TOP_K if k is None else k

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        candidate_k = self.k if k is None else k
        results = self.vector_store.similarity_search_with_relevance_scores(
            query=query,
            k=candidate_k,
        )

        documents = []
        for rank, (doc, score) in enumerate(results, start=1):
            # Copy instead of mutating an object owned by the vector store/client.
            metadata = dict(doc.metadata)
            metadata.update(
                {
                    "similarity_score": float(score),
                    "from_vector": True,
                    "vector_rank": rank,
                }
            )
            documents.append(
                Document(page_content=doc.page_content, metadata=metadata)
            )

        return documents
