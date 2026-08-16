from src.indexing.vector_store import VectorStore
from src.config.settings import settings
from src.indexing.embeddings import Embedding


class VectorRetriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store.get_vector_store()

    def retrieve(self, query: str):
        results = self.vector_store.similarity_search_with_relevance_scores(
            query=query,
            k=settings.TOP_K
        )

        documents = []

        for doc, score in results:
            doc.metadata["similarity_score"] = score
            documents.append(doc)

        return documents
